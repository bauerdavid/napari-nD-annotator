import time

import napari
import numpy as np
import cv2
from napari import Viewer
from napari.layers import Labels
from scipy.interpolate import interp1d
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QPushButton, QComboBox
from qtpy.QtCore import QThread, QObject, Signal
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from skimage.transform import SimilarityTransform, warp
from ._utils import ProgressWidget
from napari_nd_annotator._widgets._utils.persistence import PersistentWidget
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
            data = self.layer.data.copy()
            selected_label = self.layer.selected_label
            method = self.method
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                layer_slice_template = list(self.layer._slice_indices)
            prev_cnt = None
            prev_layer = None
            prev_mask = None
            start = time.time()
            for i in range(data.shape[dimension]):
                self.progress.emit(i)
                layer_slice = layer_slice_template.copy()
                layer_slice[dimension] = i
                cur_mask = data[tuple(layer_slice)] == selected_label
                cur_mask = cur_mask.astype(np.uint8)
                if cur_mask.max() == 0:
                    continue
                next_layer_slice = layer_slice.copy()
                next_layer_slice[dimension] = i + 1
                prev_layer_slice = layer_slice.copy()
                prev_layer_slice[dimension] = i - 1
                if i + 1 < data.shape[dimension] \
                        and (data[tuple(prev_layer_slice)] == selected_label).max() > 0\
                        and (data[tuple(next_layer_slice)] == selected_label).max() > 0:
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
                        cur_slice = data[tuple(inter_layer_slice)]
                        cur_slice[mask > 0] = mask[mask > 0]
                prev_cnt = cnt
                prev_layer = i
                prev_mask = cur_mask
            napari.notification_manager.receive_info("Done in %.4f s" % (time.time() - start))
            self.done.emit(data)
        except Exception as e:
            self.done.emit(None)
            raise e


class InterpolationWidget(PersistentWidget):
    def __init__(self, viewer: Viewer, parent=None):
        super().__init__("nd_annotator_interp", parent=parent)
        self.viewer = viewer
        self.progress_dialog = ProgressWidget(self, message="Interpolating slices...")
        layout = QVBoxLayout()
        self._active_labels_layer = None
        self.interpolation_thread = QThread()
        self.interpolation_worker = InterpolationWorker()
        self.interpolation_worker.moveToThread(self.interpolation_thread)
        self.interpolation_worker.done.connect(self.interpolation_thread.quit)
        self.interpolation_worker.done.connect(lambda _: self.progress_dialog.setVisible(False))
        self.interpolation_worker.done.connect(self.set_labels)
        self.interpolation_worker.progress.connect(self.progress_dialog.setValue)
        self.interpolation_thread.started.connect(self.interpolation_worker.run)
        layout.addWidget(QLabel("dimension"))
        self.dimension_dropdown = QSpinBox(self)
        self.dimension_dropdown.setMinimum(0)
        self.dimension_dropdown.setMaximum(0)
        self.dimension_dropdown.setToolTip("Dimension along which slices will be interpolated")
        layout.addWidget(self.dimension_dropdown)

        layout.addWidget(QLabel("Method", self))
        self.rpsv_widget = QWidget(self)
        self.method_dropdown = QComboBox(self)
        self.method_dropdown.addItem(CONTOUR_BASED)
        self.method_dropdown.addItem(DISTANCE_BASED)
        self.method_dropdown.addItem(RPSV)
        self.method_dropdown.setToolTip("Interpolation method\n"
                                        "%s: arithmetic mean of contour points\n"
                                        "%s: interpolation between distance maps\n"
                                        "%s: shape-aware mean of contours" % (CONTOUR_BASED, DISTANCE_BASED, RPSV))
        layout.addWidget(self.method_dropdown)
        self.add_stored_widget("method_dropdown")
        layout.addWidget(QLabel("# contour points", self))
        self.n_points = QSpinBox(self)
        self.n_points.setMinimum(10)
        self.n_points.setMaximum(1000)
        self.n_points.setToolTip("Number of contour points sampled. Used only for\"%s\" and \"%s\" methods" % (RPSV, CONTOUR_BASED))
        layout.addWidget(self.n_points)
        self.add_stored_widget("n_points")
        rpsv_layout = QVBoxLayout()
        rpsv_layout.addWidget(QLabel("max iterations", self))
        self.rpsv_iterations_spinbox = QSpinBox(self)
        self.rpsv_iterations_spinbox.setMaximum(100)
        self.rpsv_iterations_spinbox.setMinimum(1)
        self.rpsv_iterations_spinbox.setToolTip("Maximum number of iterations for RPSV. Can be fewer if points converge.")
        rpsv_layout.addWidget(self.rpsv_iterations_spinbox)
        self.add_stored_widget("rpsv_iterations_spinbox")
        self.rpsv_widget.setLayout(rpsv_layout)
        self.rpsv_widget.setVisible(self.method_dropdown.currentText() == RPSV)
        rpsv_layout.setContentsMargins(0, 0, 0, 0)
        self.rpsv_widget.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.rpsv_widget)

        self.interpolate_button = QPushButton("Interpolate", self)
        self.interpolate_button.clicked.connect(self.interpolate)
        self.interpolation_worker.done.connect(lambda: self.interpolate_button.setEnabled(True))
        layout.addWidget(self.interpolate_button)
        layout.addStretch()
        self.method_dropdown.currentTextChanged.connect(lambda _: self.rpsv_widget.setVisible(self.method_dropdown.currentText() == "RPSV"))
        self.viewer.layers.selection.events.connect(self.on_layer_selection_change)
        viewer.dims.events.order.connect(self.on_order_change)
        viewer.dims.events.ndisplay.connect(lambda _: self.interpolate_button.setEnabled(viewer.dims.ndisplay == 2))
        self.setLayout(layout)
        self.rpsv_widget.setVisible(self.method_dropdown.currentText() == "RPSV")
        self.on_order_change()
        self.interpolate_button.setEnabled(viewer.dims.ndisplay == 2)
        self.on_layer_selection_change()

    def on_layer_selection_change(self, event=None):
        if event is None or event.type == "changed":
            active_layer = self.viewer.layers.selection.active
            if isinstance(active_layer, Labels):
                self.active_labels_layer = active_layer

    @property
    def active_labels_layer(self):
        return self._active_labels_layer

    @active_labels_layer.setter
    def active_labels_layer(self, layer):
        if layer is None:
            return
        self.dimension_dropdown.setMaximum(layer.ndim)
        if self._active_labels_layer is not None:
            self._active_labels_layer.bind_key("Control-I", overwrite=True)(None)
        self._active_labels_layer = layer
        layer.bind_key("Control-I", overwrite=True)(self.interpolate)
        self.on_order_change()

    def on_order_change(self, event=None):
        if self.active_labels_layer is None or self.active_labels_layer.ndim < 3 or self.viewer.dims.ndisplay == 3:
            return
        new_dim = self.viewer.dims.not_displayed[0]
        self.dimension_dropdown.setValue(new_dim)

    def prepare_interpolation_worker(self):
        self.interpolation_worker.dimension = self.dimension_dropdown.value()
        self.interpolation_worker.n_contour_points = self.n_points.value()
        self.interpolation_worker.layer = self.active_labels_layer
        self.interpolation_worker.method = self.method_dropdown.currentText()
        self.interpolation_worker.max_iterations = self.rpsv_iterations_spinbox.value()

    def interpolate(self, _=None):
        if self.active_labels_layer is None:
            return
        self.progress_dialog.setMaximum(self.active_labels_layer.data.shape[self.dimension_dropdown.value()])
        self.progress_dialog.setVisible(True)
        self.interpolate_button.setEnabled(False)
        self.prepare_interpolation_worker()
        self.interpolation_thread.start()

    def set_labels(self, data):
        if data is None:
            return
        update_mask = data > 0
        if self.active_labels_layer.preserve_labels:
            update_mask &= self.active_labels_layer.data == 0
        self.active_labels_layer.data[update_mask] = data[update_mask]
        self.active_labels_layer.events.data()
        self.active_labels_layer.refresh()
