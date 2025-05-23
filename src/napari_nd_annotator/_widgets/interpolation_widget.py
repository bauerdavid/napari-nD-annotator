import asyncio.locks
import threading
import time
from copy import deepcopy

import napari
import numpy as np
import cv2
import scipy.ndimage
import skimage.draw
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QCursor, QPalette
from magicclass import magicclass, field, vfield, bind_key, MagicTemplate
from magicgui._util import debounce
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.qt_resources import get_current_stylesheet
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers import Labels
from napari.utils.action_manager import action_manager
from napari.qt.threading import thread_worker

from scipy.interpolate import interp1d
from qtpy.QtCore import QThread, QObject, Signal, Qt
from qtpy.QtSvg import QSvgWidget
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from skimage.transform import SimilarityTransform, warp
from ._utils import ProgressWidget
from .resources import interpolate_style_path
from .._helper_functions import layer_slice_indices, layer_dims_not_displayed, _coerce_indices_for_vectorization, \
    layer_get_order, layer_dims_displayed, layer_dims_order
from ..mean_contour import settings
from ..mean_contour.meanContour import MeanThread
from ..mean_contour._contour import calcRpsvInterpolation
from ..mean_contour._reconstruction import reconstruct
from ..mean_contour._essentials import magnitude
from .interpolation_overlay.interpolation_overlay import InterpolationOverlay, ContourList, mutex as overlay_mutex
import warnings

DISTANCE_BASED = "Distance-based"
CONTOUR_BASED = "Contour-based"
RPSV = "RPSV"

import faulthandler
faulthandler.enable()

def contour_cv2_mask_uniform(mask, contoursize_max, correct_orientation = True):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_ind = np.argmax(areas)
    contour = np.squeeze(contours[max_ind])
    contour = np.reshape(contour, (-1, 2))
    contour = np.append(contour, contour[0].reshape((-1, 2)), axis=0)
    contour = contour.astype('float32')

    rows, cols = mask.shape
    delta = np.diff(contour, axis=0)
    s = [0]
    for d in delta:
        dl = s[-1] + np.linalg.norm(d)
        s.append(dl)

    if s[-1] == 0:
        s[-1] = 1

    s = np.array(s) / s[-1]
    fx = interp1d(s, contour[:, 0] / rows, kind='linear')
    fy = interp1d(s, contour[:, 1] / cols, kind='linear')
    S = np.linspace(0, 1, contoursize_max, endpoint=False)
    X = rows * fx(S)
    Y = cols * fy(S)

    contour = np.transpose(np.stack([X, Y])).astype(np.float32)

    contour = np.stack((contour[:, 1], contour[:, 0]), axis=-1)
    if correct_orientation:
        centroid = contour.mean(0)
        start_index = np.argmin(np.abs(np.arctan2(*(contour - centroid).T)))
        contour = np.roll(contour, -start_index, 0)
    return contour


def average_mask(m1, m2, w1, w2):
    m1 = m1.astype(np.uint16)
    m2 = m2.astype(np.uint16)
    im_center = np.asarray(m1.shape)/2
    centroid1 = np.asarray(regionprops(m1)[0].centroid)
    centroid2 = np.asarray(regionprops(m2)[0].centroid)
    transl_1 = np.flip(centroid1 - im_center)
    transl_2 = np.flip(centroid2 - im_center)
    tform1 = SimilarityTransform(translation=transl_1)
    tform2 = SimilarityTransform(translation=transl_2)
    m1_translated = warp(m1, tform1, preserve_range=True).astype(bool)
    m2_translated = warp(m2, tform2, preserve_range=True).astype(bool)
    dt1_translated = distance_transform_edt(binary_erosion(m1_translated)) - distance_transform_edt(~m1_translated)
    dt2_translated = distance_transform_edt(binary_erosion(m2_translated)) - distance_transform_edt(~m2_translated)
    average_dist_translated = (w1 * dt1_translated + w2 * dt2_translated) / (w1+w2)
    average_mask_translated = average_dist_translated > 0
    transl_avg = np.flip(im_center - (centroid1*w1+centroid2*w2)/(w1+w2))
    tform_avg = SimilarityTransform(translation=transl_avg)

    return warp(average_mask_translated, tform_avg)


class MoveLoadingIconEventFilter(QObject):
    def __init__(self, icon: QSvgWidget):
        super().__init__()
        self.icon = icon
        self._is_loading = False
        self._is_above_viewer = False

    def update(self):
        if self._is_loading and self._is_above_viewer:
            self.icon.show()
        else:
            self.icon.hide()

    @property
    def is_loading(self):
        return self._is_loading

    @is_loading.setter
    def is_loading(self, value):
        self._is_loading = value
        self.update()

    def eventFilter(self, src: QObject, event: QEvent):
        if src == self.icon and event.type() == QEvent.Enter:
            pos = QCursor.pos()
        elif event.type() == QEvent.Type.HoverMove:
            pos = src.mapToGlobal(event.pos())
        elif event.type() == QEvent.Leave:
            self._is_above_viewer = False
            return False
        elif event.type() == QEvent.Enter:
            self._is_above_viewer = True
            return False
        else:
            return False
        pos.setX(pos.x() + 10)
        pos.setY(pos.y() + 10)
        self.icon.move(pos)
        self.update()
        return False


class InterpolationWorker(QObject):
    done = Signal("PyQt_PyObject")
    progress = Signal(int)
    dimension: int
    n_contour_points: int
    data: np.ndarray
    method: str
    max_iterations: int
    layer: Labels

    def run(self):
        try:
            dimension = self.dimension
            n_contour_points = self.n_contour_points
            selected_label = self.layer.selected_label
            data = (self.layer.data == selected_label).astype(np.uint8)
            method = self.method
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                layer_slice_template = list(layer_slice_indices(self.layer))
            prev_cnt = None
            prev_layer = None
            prev_mask = None
            start = time.time()
            for i in range(data.shape[dimension]):
                self.progress.emit(i)
                layer_slice = layer_slice_template.copy()
                layer_slice[dimension] = i
                cur_mask = data[tuple(layer_slice)]
                cur_mask = cur_mask.astype(np.uint8)
                if cur_mask.max() == 0:
                    continue
                next_layer_slice = layer_slice.copy()
                next_layer_slice[dimension] = i + 1
                prev_layer_slice = layer_slice.copy()
                prev_layer_slice[dimension] = i - 1
                if i + 1 < data.shape[dimension] \
                        and np.any(data[tuple(prev_layer_slice)])\
                        and np.any(data[tuple(next_layer_slice)]):
                    continue
                cnt = contour_cv2_mask_uniform(cur_mask, n_contour_points)
                if prev_cnt is None:
                    continue
                if method == RPSV:
                    stgs = settings.Settings(max_iterations=self.max_iterations,
                                             n_points=n_contour_points)
                    rpsv_thread = MeanThread([prev_cnt, cnt], stgs)
                    rpsv_thread.run()
                for j in range(prev_layer + 1, i):
                    inter_layer_slice = layer_slice_template.copy()
                    inter_layer_slice[dimension] = j
                    prev_w = i - j
                    cur_w = j - prev_layer
                    weights = [prev_w, cur_w]
                    if method == RPSV:
                        contours = rpsv_thread.contours
                        regularMean = np.zeros_like(contours[0].lookup[contours[0].parameterization, :])
                        for j in range(2):
                            regularMean += contours[j].lookup[contours[j].parameterization, :] * weights[j]
                        regularMean /= np.sum(weights)
                        q_mean = calcRpsvInterpolation(contours, weights)
                        guessRayLengths = np.zeros(contours[0].lookup[contours[0].parameterization].shape[0])
                        for i_contour in range(2):
                            contourtmp = contours[i_contour].lookup[contours[i_contour].parameterization]
                            contourlengths = magnitude(contourtmp)
                            guessRayLengths += contourlengths * weights[i_contour]
                        guessRayLengths /= np.sum(weights)
                        guessRayLengths = magnitude(regularMean)

                        qraylengths = magnitude(q_mean)
                        qraylengths[qraylengths < 1e-99] = 1e-99

                        dirs = q_mean / qraylengths.reshape(qraylengths.shape[0], 1)
                        r_mean_lengths, costs = reconstruct(q_mean, guessRayLengths.copy(), stgs, rpsv_thread.rpSignal)
                        mean_cnt = dirs * r_mean_lengths.reshape(r_mean_lengths.shape[0], 1)
                        mean_cnt = mean_cnt.astype(np.int32)
                        mask = np.zeros_like(data[tuple(inter_layer_slice)])
                        cv2.drawContours(mask, [np.flip(mean_cnt, -1)], 0, int(selected_label), -1)
                    elif method == CONTOUR_BASED:
                        mean_cnt = (prev_w * prev_cnt + cur_w * cnt)/(prev_w + cur_w)
                        mean_cnt = mean_cnt.astype(np.int32)
                        mask = np.zeros_like(data[tuple(inter_layer_slice)])
                        cv2.drawContours(mask, [np.flip(mean_cnt, -1)], 0, int(selected_label), -1)
                    elif method == DISTANCE_BASED:
                        mask = average_mask(prev_mask, cur_mask, prev_w, cur_w)
                        mask = mask.astype(np.uint8)
                        mask[mask > 0] = selected_label
                    else:
                        raise ValueError("method should be one of %s" % ((RPSV, CONTOUR_BASED, DISTANCE_BASED),))
                    data[tuple(inter_layer_slice)] = mask
                prev_cnt = cnt
                prev_layer = i
                prev_mask = cur_mask
            napari.notification_manager.receive_info("Done in %.4f s" % (time.time() - start))
            self.done.emit(data)
        except Exception as e:
            self.done.emit(None)
            raise e


@magicclass(name="Interpolation")
class InterpolationWidget(MagicTemplate):
    method = vfield(str,
                    widget_type="ComboBox",
                    options={
                        "choices": [CONTOUR_BASED, DISTANCE_BASED, RPSV],
                        "tooltip": "Interpolation method\n"
                                   f"{CONTOUR_BASED}: arithmetic mean of contour points\n"
                                   f"{DISTANCE_BASED}: interpolation between distance maps\n"
                                   f"{RPSV}: shape-aware mean of contours"
                    }
                    )
    n_contour_points = vfield(int,
                              widget_type="Slider",
                              name="Contour resolution",
                              options={
                                  "min": 10,
                                  "max": 1000,
                                  "value": 300,
                                  "tooltip": f"Number of contour points sampled. Used only for\"{RPSV}\" and \"{CONTOUR_BASED}\" methods"
                              })
    rpsv_max_iterations = field(int,
                                widget_type="SpinBox",
                                options={
                                    "min": 1,
                                    "max": 100,
                                    "value": 20,
                                    "tooltip": f"Maximum number of iterations for {RPSV}. Can be fewer if points converge."
                                })

    def __init__(self):
        self.progress_dialog = ProgressWidget(self.native, message="Interpolating slices...")
        # self.interpolation_worker.done.connect(lambda _: self.progress_dialog.setVisible(False))
        self._active_labels_layer = None
        self._is_painting = False
        self._labels_update_tmp = None
        self.workers = []
        self.workers_mutex = threading.Lock()
        self.overlay_worker = None
        self._stored_contours = []
        self.contour_mutex = overlay_mutex
        # layout.addStretch() -> how?

    def __post_init__(self):
        self._on_method_changed(self.method)
        self.native.layout().addStretch()

    def _initialize(self, viewer: napari.Viewer, annotator_widget):
        self._viewer = viewer
        self._on_active_layer_changed()
        self.viewer.dims.events.ndisplay.connect(self._ndisplay_changed)
        # self.viewer.dims.events.ndisplay.connect(lambda _: self._reset_contours() if self.viewer.dims.ndisplay == 3)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)
        action_manager.register_action("napari-nD-annotator:interpolate", self.Interpolate,
                                       "Interpolate missing slices (I)", None)
        self._annotator_widget = annotator_widget

        self.viewer.dims.events.current_step.connect(self._current_step_changed)
        self.viewer.dims.events.order.connect(self._dims_order_changed)
        self.loading_icon = QSvgWidget("C:\\Users\\User\\Downloads\\loading.svg", self._viewer.window._qt_window)
        self.loading_icon.setStyleSheet("background: transparent;")
        self.loading_icon.setAutoFillBackground(False)
        palette = self.loading_icon.palette()
        palette.setBrush(QPalette.Base, Qt.transparent)
        self.loading_icon.setPalette(palette)
        self.loading_icon.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.loading_icon.setFixedSize(20, 20)
        self.loading_icon.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.loading_icon.setAttribute(Qt.WA_ShowWithoutActivating)
        self.loading_icon.setAttribute(Qt.WA_TranslucentBackground)

        self.loading_icon.show()
        self.event_filter = MoveLoadingIconEventFilter(self.loading_icon)
        self.loading_icon.installEventFilter(self.event_filter)
        self.viewer.window._qt_window.installEventFilter(self.event_filter)
        self.native.setMouseTracking(True)

    @debounce
    def _current_step_changed(self, _=None):
        if self.active_labels_layer is None or "interpolation" not in self.active_labels_layer._overlays:
            return
        if self.overlay_worker is not None:
            self.overlay_worker.quit()
        self._update_overlay()

    def _dims_order_changed(self):
        self._interpolate_all()

    def _ndisplay_changed(self):
        is_2d = self.viewer.dims.ndisplay == 2
        self.interpolate_button.options.update(enabled=is_2d)
        if is_2d:
            self._interpolate_all()
        else:
            self._reset_contours()

    # @thread_worker
    def _update_overlay(self):
        if self._stored_contours is None:
            contour = []
        else:
            contour = deepcopy(self._stored_contours[self.current_slice])
        with self.contour_mutex:
            self.active_labels_layer._overlays["interpolation"].contour = ContourList(contour)

    @property
    def viewer(self):
        return self._viewer

    @property
    def current_slice(self):
        if self.dimension is None:
            return None
        return layer_slice_indices(self.active_labels_layer)[self.dimension]

    def Interpolate(self, _=None):
        if self.active_labels_layer is None or self.dimension is None:
            return
        # self.progress_dialog.setMaximum(self.active_labels_layer.data.shape[self.dimension])
        # self.progress_dialog.setVisible(True)
        worker = self._store_interpolation()
        def on_worker_finished():
            self.interpolate_button.enabled = True
            self.event_filter.is_loading = False
        worker.finished.connect(on_worker_finished)
        self.interpolate_button.enabled = False
        worker.start()
        self.event_filter.is_loading = True

    @thread_worker
    def _store_interpolation(self):
        if self.active_labels_layer is None or "interpolation" not in self.active_labels_layer._overlays:
            return
        for i in range(len(self._stored_contours)):
            with self.contour_mutex:
                slice_contours = self._stored_contours[i]
            slice_template = list(layer_slice_indices(self.active_labels_layer))
            cur_slice = slice_template.copy()
            cur_slice[self.dimension] = i
            cur_slice = tuple(cur_slice)
            for contour in slice_contours:
                mask_slice = self.active_labels_layer.data[cur_slice]
                rr, cc = skimage.draw.polygon(contour[:, 1], contour[:, 0], mask_slice.shape)
                mask_slice[rr, cc] = self.active_labels_layer.selected_label
            with self.contour_mutex:
                self._stored_contours[i] = []

    def _interpolate_all(self):
        if self.active_labels_layer is None or "interpolation" not in self.active_labels_layer._overlays:
            return
        slices_with_data = np.argwhere(np.any(
            self.active_labels_layer.data == self.active_labels_layer.selected_label,
            axis=tuple(layer_dims_displayed(self.active_labels_layer))
        ))
        self._reset_contours()
        slices_with_data = np.squeeze(slices_with_data)
        for idx in slices_with_data:
            self._start_interpolation(idx, 1)
        self._update_overlay()

    @thread_worker
    def _interpolate_from_slice(self, idx, direction):
        assert direction in [-1, 1], "direction should be either -1 or 1"
        if self.active_labels_layer is None:
            return

        dimension = self.dimension
        n_contour_points = self.n_contour_points
        selected_label = self.active_labels_layer.selected_label
        data = (self.active_labels_layer.data == selected_label).astype(np.uint8)
        method = self.method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layer_slice_template = list(layer_slice_indices(self.active_labels_layer))
        cur_slice = layer_slice_template.copy()
        cur_slice[dimension] = idx
        cur_mask = data[tuple(cur_slice)]
        cur_mask = cur_mask.astype(np.uint8)
        if cur_mask.max() == 0:
            return
        curr_cnt = contour_cv2_mask_uniform(cur_mask, n_contour_points)
        start_idx = idx + direction
        end_idx = -1 if direction == -1 else data.shape[dimension]
        for i in range(start_idx, end_idx, direction):
            layer_slice = layer_slice_template.copy()
            layer_slice[dimension] = i
            next_mask = data[tuple(layer_slice)]
            next_mask = next_mask.astype(np.uint8)
            if next_mask.max() == 0:
                continue
            if abs(i-idx) == 1:
                break
            next_cnt = contour_cv2_mask_uniform(next_mask, n_contour_points)
            if method == RPSV:
                stgs = settings.Settings(max_iterations=self.rpsv_max_iterations.value,
                                         n_points=n_contour_points)
                rpsv_thread = MeanThread([curr_cnt, next_cnt], stgs)
                rpsv_thread.run()
            for j in range(start_idx, i, direction):
                inter_layer_slice = layer_slice_template.copy()
                inter_layer_slice[dimension] = j
                prev_w = abs(i - j)
                cur_w = abs(j - idx)
                weights = [prev_w, cur_w]
                if method == RPSV:
                    contours = rpsv_thread.contours
                    regularMean = np.zeros_like(contours[0].lookup[contours[0].parameterization, :])
                    for k in range(2):
                        regularMean += contours[k].lookup[contours[k].parameterization, :] * weights[k]
                    regularMean /= np.sum(weights)
                    q_mean = calcRpsvInterpolation(contours, weights)
                    guessRayLengths = np.zeros(contours[0].lookup[contours[0].parameterization].shape[0])
                    for i_contour in range(2):
                        contourtmp = contours[i_contour].lookup[contours[i_contour].parameterization]
                        contourlengths = magnitude(contourtmp)
                        guessRayLengths += contourlengths * weights[i_contour]
                    guessRayLengths /= np.sum(weights)
                    guessRayLengths = magnitude(regularMean)

                    qraylengths = magnitude(q_mean)
                    qraylengths[qraylengths < 1e-99] = 1e-99

                    dirs = q_mean / qraylengths.reshape(qraylengths.shape[0], 1)
                    r_mean_lengths, costs = reconstruct(q_mean, guessRayLengths.copy(), stgs, rpsv_thread.rpSignal)
                    mean_cnt = dirs * r_mean_lengths.reshape(r_mean_lengths.shape[0], 1)
                    mean_cnt = mean_cnt.astype(np.int32)
                    mean_cnt = [mean_cnt]
                    # mask = np.zeros_like(data[tuple(inter_layer_slice)])
                    # cv2.drawContours(mask, [np.flip(mean_cnt, -1)], 0, int(selected_label), -1)
                elif method == CONTOUR_BASED:
                    mean_cnt = (prev_w * curr_cnt + cur_w * next_cnt) / (prev_w + cur_w)
                    mean_cnt = mean_cnt.astype(np.int32)
                    mean_cnt = [mean_cnt]
                    # mask = np.zeros_like(data[tuple(inter_layer_slice)])
                    # cv2.drawContours(mask, [np.flip(mean_cnt, -1)], 0, int(selected_label), -1)
                elif method == DISTANCE_BASED:
                    mask = average_mask(cur_mask, next_mask, prev_w, cur_w)
                    mask, n_labels = scipy.ndimage.label(mask)
                    mean_cnt = []
                    for label in range(1, n_labels+1):
                        mean_cnt.append(contour_cv2_mask_uniform(mask == label, n_contour_points))
                    # mask[mask > 0] = selected_label
                else:
                    raise ValueError("method should be one of %s" % ((RPSV, CONTOUR_BASED, DISTANCE_BASED),))
                yield j, mean_cnt
            break
        with self.contour_mutex:
            self._stored_contours[idx] = []

    def _execute_interpolation(self, idx):
        for direction in [-1, 1]:
            self._start_interpolation(idx, direction)

    def _start_interpolation(self, idx, direction):
        worker = self._interpolate_from_slice(idx, direction)
        with self.workers_mutex:
            self.workers.append(worker)
        worker.yielded.connect(self._store_contour)
        def remove_worker():
            with self.workers_mutex:
                self.workers[self.workers.index(worker)] = None
        worker.finished.connect(remove_worker)
        worker.start()

    @method.connect
    def _on_method_changed(self, new_method):
        self.rpsv_max_iterations.visible = new_method == RPSV
        self._interpolate_all()

    @property
    def active_labels_layer(self) -> Labels | None:
        return self._active_labels_layer

    @active_labels_layer.setter
    def active_labels_layer(self, layer):
        if layer == self._active_labels_layer:
            return
        if self._active_labels_layer is not None and self._on_mouse_drag in self._active_labels_layer.mouse_drag_callbacks:
            self._active_labels_layer.mouse_drag_callbacks.remove(self._on_mouse_drag)
            self._active_labels_layer.events.labels_update.disconnect(self._on_labels_update)
            self._active_labels_layer.events.selected_label.disconnect(self._interpolate_all)
            if "interpolation" in self._active_labels_layer._overlays:
                self._active_labels_layer._overlays["interpolation"].contour = ContourList()
        self._active_labels_layer = layer if isinstance(layer, Labels) and layer.ndim == 3 else None
        if self._active_labels_layer is None or self._active_labels_layer.ndim < 3:
            self._reset_contours()
            return
        if self._on_mouse_drag not in self._active_labels_layer.mouse_drag_callbacks:
            self._active_labels_layer.mouse_drag_callbacks.append(self._on_mouse_drag)
            self._active_labels_layer.events.labels_update.connect(self._on_labels_update)
            self._active_labels_layer.events.selected_label.connect(self._interpolate_all)

        if "interpolation" not in self._active_labels_layer._overlays:
            interpolation_overlay = InterpolationOverlay()
            self._active_labels_layer._overlays.update({"interpolation": interpolation_overlay})
            labels_control: QtLabelsControls = self.viewer.window.qt_viewer.controls.widgets[self._active_labels_layer]
            if labels_control.button_grid.itemAtPosition(1, 1) is not None:
                return
            interpolate_button = QtModePushButton(
                self._active_labels_layer,
                'interpolate',
            )
            action_manager.bind_button(
                'napari-nD-annotator:interpolate', interpolate_button
            )
            interpolate_button.setStyleSheet(get_current_stylesheet([interpolate_style_path]))
            labels_control.button_group.addButton(interpolate_button)
            labels_control.interpolate_button = interpolate_button
            labels_control.button_grid.addWidget(labels_control.interpolate_button, 1, 1)
            self._active_labels_layer.bind_key("I", self.Interpolate)
        self._reset_contours()
        self._interpolate_all()

    @property
    def interpolate_button(self):
        for widget in self._list:
            if widget.name == "Interpolate":
                return widget

    def _enable_interpolation_button(self, *_):
        self.interpolate_button.enabled = True

    def _on_active_layer_changed(self, *_):
        self.active_labels_layer = self.viewer.layers.selection.active
        if self.active_labels_layer is None or self.active_labels_layer.ndim == 2:
            self.interpolate_button.enabled = False
            return
        self.interpolate_button.enabled = True

    @property
    def dimension(self):
        if self.active_labels_layer is None:
            return None
        dims_not_displayed = layer_dims_not_displayed(self.active_labels_layer)
        return None if len(dims_not_displayed) != 1 else dims_not_displayed[0]

    def _on_mouse_drag(self, layer: Labels, event):
        if (layer.mode not in ["erase", "paint"]
                and ("minimal_contour" not in layer._overlays or not layer._overlays["minimal_contour"].enabled)):
            return
        self._is_painting = True
        if layer.mode in ["erase", "paint"]:
            cur_slice = layer_slice_indices(layer)
            with self.contour_mutex:
                cnt_list = self._stored_contours[cur_slice[self.dimension]]
            if len(cnt_list) > 0:
                mask = np.zeros_like(layer.data[cur_slice])
                for i in range(len(cnt_list)):
                    cnt_list[i] = np.asarray(cnt_list[i], dtype=np.int32)

                cv2.drawContours(mask, cnt_list, -1, 1, -1)
                update_idx = np.nonzero(mask)
                order = layer_dims_order(layer)
                extended_idx = tuple(update_idx[i-1] if i in layer_dims_displayed(layer) else np.full_like(update_idx[0], cur_slice[order[i]]) for i in range(layer.ndim))
                extended_idx = _coerce_indices_for_vectorization(self.active_labels_layer.data, extended_idx)
                layer.data_setitem(extended_idx, layer.selected_label)
                with self.contour_mutex:
                    layer._overlays["interpolation"].contour = ContourList()


        yield
        while event.type == "mouse_move":
            yield
        self._is_painting = False
        if self._labels_update_tmp is not None:
            data, offset = self._labels_update_tmp
            self._labels_update_tmp = None
            if layer.mode == "erase" or not self._annotator_widget.autofill_objects:
                self._active_labels_layer.events.labels_update(data=data, offset=offset)
            cur_slice_data = layer.data[layer_slice_indices(layer)]
            if not np.any(cur_slice_data):
                self._interpolate_all()

    def _on_labels_update(self, event):
        if self._is_painting:
            self._labels_update_tmp = (event.data, event.offset)
            return
        self._execute_interpolation(layer_slice_indices(self._active_labels_layer)[self.dimension])

    def _on_set_data(self, event):
        ...

    def _store_contour(self, data):
        idx, contour_list = data
        uniques = np.empty(self.n_contour_points, dtype=bool)
        for i, contour in enumerate(contour_list):

            int_cnt = np.round(contour).astype(int)
            uniques[:-1] = np.any(int_cnt[:-1] != int_cnt[1:], axis=1)
            uniques[-1] = True
            contour = contour[uniques]
            contour = np.fliplr(contour)
            contour_list[i] = contour
        with self.contour_mutex:
            self._stored_contours[idx] = contour_list
        if idx == self.current_slice:
            with self.contour_mutex:
                self.active_labels_layer._overlays["interpolation"].contour = ContourList(contour_list)

    def _reset_contours(self):
        if self.dimension is None:
            with self.contour_mutex:
                self._stored_contours = None
            return
        with self.contour_mutex:
            self._stored_contours = [[] for _ in range(self.active_labels_layer.data.shape[self.dimension])]
            if self.active_labels_layer is not None:
                self.active_labels_layer._overlays["interpolation"].contour = ContourList()

    def _reset_interpolation_worker(self):
        self.interpolation_worker = None

    def _reset_overlay_worker(self):
        self.interpolation_worker = None