import numpy as np
import cv2
from napari import Viewer
from napari.layers import Labels
from scipy.interpolate import interp1d
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QPushButton

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

        layout.addWidget(QLabel("# contour points"))
        self.n_points = QSpinBox()
        self.n_points.setMinimum(10)
        self.n_points.setMaximum(1000)
        self.n_points.setValue(300)
        layout.addWidget(self.n_points)

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
        layer_slice_template = [
            slice(None) if d in self.viewer.dims.displayed
                else None if d == dimension
                else self.viewer.dims.current_step[d]
            for d in range(self.active_labels_layer.ndim)]
        prev_cnt = None
        prev_layer = None
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
                for j in range(prev_layer + 1, i):
                    inter_layer_slice = layer_slice_template.copy()
                    inter_layer_slice[dimension] = j
                    prev_w = i - j
                    cur_w = j - prev_layer
                    mean_cnt = (prev_w * prev_cnt + cur_w * cnt)/(prev_w + cur_w)
                    mean_cnt = mean_cnt.astype(np.int32)
                    mask = np.zeros_like(data[tuple(inter_layer_slice)])
                    cv2.drawContours(mask, [np.flip(mean_cnt, -1)], 0, self.active_labels_layer.selected_label, -1)
                    cur_slice = data[tuple(inter_layer_slice)]
                    cur_slice[mask > 0] = mask[mask > 0]
            prev_cnt = cnt
            prev_layer = i
            self.active_labels_layer.refresh()
