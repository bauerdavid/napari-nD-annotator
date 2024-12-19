
import time

import napari
import numpy as np
import cv2
from magicclass import magicclass, field, vfield, bind_key, MagicTemplate
from napari.layers import Labels

from scipy.interpolate import interp1d
from qtpy.QtCore import QThread, QObject, Signal
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from skimage.transform import SimilarityTransform, warp
from ._utils import ProgressWidget
from .._helper_functions import layer_slice_indices, layer_dims_not_displayed, _coerce_indices_for_vectorization
from ..mean_contour import settings
from ..mean_contour.meanContour import MeanThread
from ..mean_contour._contour import calcRpsvInterpolation
from ..mean_contour._reconstruction import reconstruct
from ..mean_contour._essentials import magnitude
import warnings

DISTANCE_BASED = "Distance-based"
CONTOUR_BASED = "Contour-based"
RPSV = "RPSV"


def contour_cv2_mask_uniform(mask, contoursize_max):
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
                centroid = cnt.mean(0)
                start_index = np.argmin(np.abs(np.arctan2(*(cnt - centroid).T)))
                cnt = np.roll(cnt, -start_index, 0)
                if prev_cnt is not None:
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
        self.interpolation_worker = InterpolationWorker()
        self.interpolation_worker.moveToThread(self.interpolation_thread)
        self.interpolation_worker.done.connect(self.interpolation_thread.quit)
        self.interpolation_worker.done.connect(lambda _: self.progress_dialog.setVisible(False))
        self.interpolation_worker.done.connect(self._set_labels)
        self.interpolation_worker.progress.connect(self.progress_dialog.setValue)
        self.interpolation_thread.started.connect(self.interpolation_worker.run)
        self.interpolation_worker.done.connect(self._enable_interpolation_button)
        # layout.addStretch() -> how?

    def __post_init__(self):
        self._on_method_changed(self.method)
        self.native.layout().addStretch()

    def _initialize(self, viewer: napari.Viewer):
        self._viewer = viewer
        self._viewer.bind_key("Ctrl-I")(self.Interpolate)
        self._on_active_layer_changed()
        self.viewer.dims.events.ndisplay.connect(
            lambda _: self.interpolate_button.options.update(enabled=self.viewer.dims.ndisplay == 2))
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

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
        if self.active_labels_layer is None:
            return
        self.progress_dialog.setMaximum(self.active_labels_layer.data.shape[self.dimension])
        self.progress_dialog.setVisible(True)
        self.interpolate_button.enabled = False
        self._prepare_interpolation_worker()
        self.interpolation_thread.start()

    @method.connect
    def _on_method_changed(self, new_method):
        self.rpsv_max_iterations.visible = new_method == RPSV

    def _set_labels(self, update_mask):
        if update_mask is None:
            return
        update_val = self.active_labels_layer.selected_label
        if self.active_labels_layer.preserve_labels:
            update_mask = np.logical_and(update_mask, self.active_labels_layer.data == 0)
        update_idx = np.nonzero(update_mask)
        update_idx = _coerce_indices_for_vectorization(self.active_labels_layer.data, update_idx)
        if self.active_labels_layer.preserve_labels:
            update_mask &= self.active_labels_layer.data == 0
        if hasattr(self.active_labels_layer, "data_setitem"):
            self.active_labels_layer.data_setitem(update_idx, update_val)
        else:
            self.active_labels_layer._save_history((update_idx, self.active_labels_layer.data[update_mask], update_val))
            self.active_labels_layer.data[update_mask] = update_val
            self.active_labels_layer.events.data()
            self.active_labels_layer.refresh()

    @property
    def active_labels_layer(self) -> Labels | None:
        active_layer = self.viewer.layers.selection.active
        return active_layer if isinstance(active_layer, Labels) else None

    @property
    def interpolate_button(self):
        for widget in self._list:
            if widget.name == "Interpolate":
                return widget

    def _enable_interpolation_button(self, *_):
        self.interpolate_button.enabled = True

    def _on_active_layer_changed(self, *_):
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
