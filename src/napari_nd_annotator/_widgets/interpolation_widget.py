
import time

import napari
import numpy as np
import cv2
from magicclass import magicclass, field, vfield, bind_key, MagicTemplate
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.qt_resources import get_current_stylesheet
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers import Labels
from napari.utils.action_manager import action_manager
from napari.qt.threading import thread_worker

from scipy.interpolate import interp1d
from qtpy.QtCore import QThread, QObject, Signal
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
from .interpolation_overlay.interpolation_overlay import InterpolationOverlay
import warnings

DISTANCE_BASED = "Distance-based"
CONTOUR_BASED = "Contour-based"
RPSV = "RPSV"


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
    transl_1 = centroid1 - im_center
    transl_2 = centroid2 - im_center
    tform1 = SimilarityTransform(translation=np.flip(transl_1))
    tform2 = SimilarityTransform(translation=np.flip(transl_2))
    m1_translated = warp(m1, tform1, preserve_range=True).astype(bool)
    m2_translated = warp(m2, tform2, preserve_range=True).astype(bool)
    dt1_translated = distance_transform_edt(binary_erosion(m1_translated)) - distance_transform_edt(~m1_translated)
    dt2_translated = distance_transform_edt(binary_erosion(m2_translated)) - distance_transform_edt(~m2_translated)
    average_dist_translated = (w1 * dt1_translated + w2 * dt2_translated) / (w1+w2)
    average_mask_translated = average_dist_translated > 0
    transl_avg = im_center - (centroid1*w1+centroid2*w2)/(w1+w2)
    tform_avg = SimilarityTransform(translation=np.flip(transl_avg))

    return warp(average_mask_translated, tform_avg)


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
        self.interpolation_thread = QThread()
        self.interpolation_worker = None
        # self.interpolation_worker.moveToThread(self.interpolation_thread)
        # self.interpolation_worker.done.connect(self.interpolation_thread.quit)
        # self.interpolation_worker.done.connect(lambda _: self.progress_dialog.setVisible(False))
        # self.interpolation_worker.done.connect(self._set_labels)
        # self.interpolation_worker.progress.connect(self.progress_dialog.setValue)
        # self.interpolation_thread.started.connect(self.interpolation_worker.run)
        # self.interpolation_worker.done.connect(self._enable_interpolation_button)
        self._active_labels_layer = None
        self._is_painting = False
        self._labels_update_tmp = None
        # layout.addStretch() -> how?

    def __post_init__(self):
        self._on_method_changed(self.method)
        self.native.layout().addStretch()

    def _initialize(self, viewer: napari.Viewer, annotator_widget):
        self._viewer = viewer
        self._on_active_layer_changed()
        self.viewer.dims.events.ndisplay.connect(
            lambda _: self.interpolate_button.options.update(enabled=self.viewer.dims.ndisplay == 2))
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)
        action_manager.register_action("napari-nD-annotator:interpolate", self.Interpolate,
                                       "Interpolate missing slices (I)", None)
        self._annotator_widget = annotator_widget

        def current_step_changed(_=None):
            if self.active_labels_layer is None or "interpolation" not in self.active_labels_layer._overlays:
                return
            self.active_labels_layer._overlays["interpolation"].current_slice = layer_slice_indices(self.active_labels_layer)[self.dimension]

        self.viewer.dims.events.current_step.connect(current_step_changed)

    @property
    def viewer(self):
        return self._viewer

    def _prepare_interpolation_worker(self):
        self.interpolation_worker.dimension = self.dimension
        self.interpolation_worker.n_contour_points = self.n_contour_points
        self.interpolation_worker.layer = self.active_labels_layer
        self.interpolation_worker.method = self.method
        self.interpolation_worker.max_iterations = self.rpsv_max_iterations.value

    def Interpolate(self, _=None):
        if self.active_labels_layer is None or self.dimension is None:
            return
        self.progress_dialog.setMaximum(self.active_labels_layer.data.shape[self.dimension])
        self.progress_dialog.setVisible(True)
        self.interpolate_button.enabled = False
        self._prepare_interpolation_worker()
        self.interpolation_thread.start()

    @thread_worker
    def _execute_interpolation(self, idx):
        if self.active_labels_layer is None:
            return
        print(f"processing {idx}")
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
        for start_idx, end_idx, step in [(idx-1, -1, -1), (idx+1, data.shape[dimension], 1)]:
            for i in range(start_idx, end_idx, step):
                print(f"slice {i}")
                layer_slice = layer_slice_template.copy()
                layer_slice[dimension] = i
                next_mask = data[tuple(layer_slice)]
                next_mask = next_mask.astype(np.uint8)
                if next_mask.max() == 0:
                    continue
                if abs(i-idx) == 1:
                    print("neighboring annotations")
                    break
                print(f"next contour at {i} (direction: {step})")
                next_cnt = contour_cv2_mask_uniform(next_mask, n_contour_points)
                if method == RPSV:
                    stgs = settings.Settings(max_iterations=self.max_iterations,
                                             n_points=n_contour_points)
                    rpsv_thread = MeanThread([curr_cnt, next_cnt], stgs)
                    rpsv_thread.run()
                for j in range(start_idx, i, step):
                    inter_layer_slice = layer_slice_template.copy()
                    inter_layer_slice[dimension] = j
                    prev_w = abs(i - j)
                    cur_w = abs(j - idx)
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
                        # mask = np.zeros_like(data[tuple(inter_layer_slice)])
                        # cv2.drawContours(mask, [np.flip(mean_cnt, -1)], 0, int(selected_label), -1)
                    elif method == CONTOUR_BASED:
                        mean_cnt = (prev_w * curr_cnt + cur_w * next_cnt) / (prev_w + cur_w)
                        mean_cnt = mean_cnt.astype(np.int32)
                        # mask = np.zeros_like(data[tuple(inter_layer_slice)])
                        # cv2.drawContours(mask, [np.flip(mean_cnt, -1)], 0, int(selected_label), -1)
                    elif method == DISTANCE_BASED:
                        mask = average_mask(cur_mask, next_mask, prev_w, cur_w)
                        mask = mask.astype(np.uint8)
                        mean_cnt = contour_cv2_mask_uniform(mask, n_contour_points)
                        # mask[mask > 0] = selected_label
                    else:
                        raise ValueError("method should be one of %s" % ((RPSV, CONTOUR_BASED, DISTANCE_BASED),))
                    yield j, mean_cnt
                break
        self.active_labels_layer._overlays["interpolation"].points_per_slice[idx] = None

    @method.connect
    def _on_method_changed(self, new_method):
        self.rpsv_max_iterations.visible = new_method == RPSV

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
        self._active_labels_layer = layer if isinstance(layer, Labels) else None
        if self._active_labels_layer is not None and self._on_mouse_drag not in self._active_labels_layer.mouse_drag_callbacks:
            self._active_labels_layer.mouse_drag_callbacks.append(self._on_mouse_drag)
            self._active_labels_layer.events.labels_update.connect(self._on_labels_update)
        if self._active_labels_layer is None or self._active_labels_layer.ndim < 3:
            return
        if "interpolation" not in self._active_labels_layer._overlays:
            interpolation_overlay = InterpolationOverlay()
            self._active_labels_layer._overlays.update({"interpolation": interpolation_overlay})
            interpolation_overlay.points_per_slice = [None] * self._active_labels_layer.data.shape[self.dimension]
            interpolation_overlay.current_slice = layer_slice_indices(self._active_labels_layer)[self.dimension]
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
        return None if len(dims_not_displayed) == 0 else dims_not_displayed[0]

    def _on_mouse_drag(self, layer: Labels, event):
        if layer.mode not in ["erase", "paint"] and not layer._overlays["minimal_contour"].enabled:
            return
        self._is_painting = True
        if layer.mode in ["erase", "paint"]:
            cur_slice = layer_slice_indices(layer)
            cnt = layer._overlays["interpolation"].points_per_slice[cur_slice[self.dimension]]
            if cnt is not None:
                cnt = np.asarray(cnt)
                mask = np.zeros_like(layer.data[cur_slice])
                cv2.drawContours(mask, [cnt], 0, 1, -1)
                update_idx = np.nonzero(mask)
                order = layer_dims_order(layer)
                extended_idx = tuple(update_idx[i-1] if i in layer_dims_displayed(layer) else np.full_like(update_idx[0], cur_slice[order[i]]) for i in range(layer.ndim))
                extended_idx = _coerce_indices_for_vectorization(self.active_labels_layer.data, extended_idx)
                layer.data_setitem(extended_idx, layer.selected_label)
                layer._overlays["interpolation"].points_per_slice[cur_slice[self.dimension]] = None
                layer._overlays["interpolation"].events.points_per_slice()


        yield
        while event.type == "mouse_move":
            yield
        self._is_painting = False
        if self._labels_update_tmp is not None:
            data, offset = self._labels_update_tmp
            self._labels_update_tmp = None
            if layer.mode == "erase" or not self._annotator_widget.autofill_objects:
                self._active_labels_layer.events.labels_update(data=data, offset=offset)

    def _on_labels_update(self, event):
        if self._is_painting:
            self._labels_update_tmp = (event.data, event.offset)
            return
        if self.interpolation_worker is not None:
            self.interpolation_worker.quit()
        self.interpolation_worker = self._execute_interpolation(layer_slice_indices(self._active_labels_layer)[self.dimension])
        self.interpolation_worker.yielded.connect(self._store_contour)
        self.interpolation_worker.finished.connect(self._reset_interpolation_worker)
        self.interpolation_worker.start()

    def _store_contour(self, data):
        idx, contour = data
        uniques = np.empty(len(contour), dtype=bool)
        uniques[:-1] = np.any(contour[:-1] != contour[1:], axis=1)
        uniques[-1] = True
        contour = contour[uniques]
        self.active_labels_layer._overlays["interpolation"].points_per_slice[idx] = list(np.fliplr(contour))
        if idx == self.active_labels_layer._overlays["interpolation"].current_slice:
            self.active_labels_layer._overlays["interpolation"].events.points_per_slice()

    def _reset_interpolation_worker(self):
        self.interpolation_worker = None
