import time

import napari
import numpy as np
import cv2
from napari import Viewer
from napari.layers import Labels
from scipy.interpolate import interp1d
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QPushButton, QComboBox

from ..mean_contour import settings
from ..mean_contour.meanContour import MeanThread
from ..mean_contour._contour import calcRpsvInterpolation
from ..mean_contour._reconstruction import reconstruct
from ..mean_contour._essentials import magnitude


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

    if (s[-1] == 0):
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


class InterpolationWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        layout = QVBoxLayout()
        self.active_labels_layer = None

        layout.addWidget(QLabel("dimension"))
        self.dimension_dropdown = QSpinBox()
        self.dimension_dropdown.setMinimum(0)
        self.dimension_dropdown.setMaximum(0)
        layout.addWidget(self.dimension_dropdown)

        layout.addWidget(QLabel("Method"))
        self.method_dropdown = QComboBox()
        self.method_dropdown.addItem("Geometric mean")
        self.method_dropdown.addItem("RPSV")
        self.method_dropdown.currentTextChanged.connect(lambda _: self.rpsv_widget.setVisible(self.method_dropdown.currentText() == "RPSV"))
        layout.addWidget(self.method_dropdown)

        layout.addWidget(QLabel("# contour points"))
        self.n_points = QSpinBox()
        self.n_points.setMinimum(10)
        self.n_points.setMaximum(1000)
        self.n_points.setValue(300)
        layout.addWidget(self.n_points)

        self.rpsv_widget = QWidget()
        rpsv_layout = QVBoxLayout()
        rpsv_layout.addWidget(QLabel("max iterations"))
        self.rpsv_iterations_spinbox = QSpinBox()
        self.rpsv_iterations_spinbox.setMaximum(100)
        self.rpsv_iterations_spinbox.setMinimum(1)
        self.rpsv_iterations_spinbox.setValue(20)
        rpsv_layout.addWidget(self.rpsv_iterations_spinbox)
        self.rpsv_widget.setLayout(rpsv_layout)
        self.rpsv_widget.setVisible(self.method_dropdown.currentText() == "RPSV")
        rpsv_layout.setContentsMargins(0, 0, 0, 0)
        self.rpsv_widget.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.rpsv_widget)


        self.interpolate_button = QPushButton("Interpolate")
        self.interpolate_button.clicked.connect(self.interpolate)
        layout.addWidget(self.interpolate_button)
        self.viewer.layers.selection.events.connect(self.on_layer_selection_change)
        viewer.dims.events.order.connect(self.on_order_change)
        viewer.dims.events.ndisplay.connect(lambda _: self.interpolate_button.setEnabled(viewer.dims.ndisplay == 2))
        self.setLayout(layout)
        layout.addStretch()
        self.on_layer_selection_change()

    def on_layer_selection_change(self, event=None):
        if event is None or event.type == "changed":
            active_layer = self.viewer.layers.selection.active
            if isinstance(active_layer, Labels):
                self.active_labels_layer = active_layer
                self.dimension_dropdown.setMaximum(self.active_labels_layer.ndim)

    def set_has_channels(self, has_channels):
        self.channels_dim_dropdown.setEnabled(has_channels)

    def on_order_change(self, event):
        extents = self.active_labels_layer.extent.data[1] + 1
        not_displayed = extents[list(self.viewer.dims.not_displayed)]
        ch_excluded = list(filter(lambda x: x > 3, not_displayed))
        if len(ch_excluded) == 0:
            new_dim = self.viewer.dims.not_displayed[0]
        else:
            new_dim = self.viewer.dims.order[int(np.argwhere(not_displayed == ch_excluded[0])[0])]
        self.dimension_dropdown.setValue(new_dim)

    def interpolate(self):
        if self.active_labels_layer is None:
            return
        dimension = self.dimension_dropdown.value()
        n_contour_points = self.n_points.value()
        data = self.active_labels_layer.data
        use_rpsv = self.method_dropdown.currentText() == "RPSV"
        layer_slice_template = [
            slice(None) if d in self.viewer.dims.displayed
                else None if d == dimension
                else self.viewer.dims.current_step[d]
            for d in range(self.active_labels_layer.ndim)]
        prev_cnt = None
        prev_layer = None
        start = time.time()
        for i in range(data.shape[dimension]):
            layer_slice = layer_slice_template.copy()
            layer_slice[dimension] = i
            mask = data[tuple(layer_slice)] == self.active_labels_layer.selected_label
            mask = mask.astype(np.uint8)
            if mask.max() == 0:
                continue
            cnt = contour_cv2_mask_uniform(mask, n_contour_points)
            centroid = cnt.mean(0)
            start_index = np.argmin(np.abs(np.arctan2(*(cnt - centroid).T)))
            cnt = np.roll(cnt, -start_index, 0)
            if prev_cnt is not None:
                if use_rpsv:
                    stgs = settings.Settings(max_iterations=self.rpsv_iterations_spinbox.value(),
                                             n_points=n_contour_points)
                    rpsv_thread = MeanThread([prev_cnt, cnt], stgs)
                    rpsv_thread.run()
                for j in range(prev_layer + 1, i):
                    inter_layer_slice = layer_slice_template.copy()
                    inter_layer_slice[dimension] = j
                    prev_w = i - j
                    cur_w = j - prev_layer
                    weights = [prev_w, cur_w]
                    if use_rpsv:
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
                        print(mean_cnt.shape, mean_cnt.min(), mean_cnt.max())
                    else:
                        mean_cnt = (prev_w * prev_cnt + cur_w * cnt)/(prev_w + cur_w)
                    mean_cnt = mean_cnt.astype(np.int32)
                    mask = np.zeros_like(data[tuple(inter_layer_slice)])
                    cv2.drawContours(mask, [np.flip(mean_cnt, -1)], 0, self.active_labels_layer.selected_label, -1)
                    cur_slice = data[tuple(inter_layer_slice)]
                    cur_slice[mask > 0] = mask[mask > 0]
            prev_cnt = cnt
            prev_layer = i
            self.active_labels_layer.refresh()
        napari.notification_manager.receive_info("Done in %.4f s" % (time.time() - start))
