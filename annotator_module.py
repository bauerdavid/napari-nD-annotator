from cv2 import cv2
from PyQt5.QtCore import QObject, QEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QTabWidget, QLabel, QComboBox, QSpinBox, QPushButton
from napari import Viewer
from napari.layers import Labels, Layer
import numpy as np
from scipy.interpolate import interp1d
from skimage.draw import draw
from scipy.ndimage import binary_fill_holes
import math


def generate_label_colors(n):
    import colorsys
    from random import shuffle, seed
    rgb_tuples = [colorsys.hsv_to_rgb(x * 1.0 / n, 1., 1.) for x in range(n)]
    rgb_tuples = [(int(rgb_tuples[i][0] * 255), int(rgb_tuples[i][1] * 255), int(rgb_tuples[i][2] * 255)) for i in
                  range(n)]
    seed(0)
    shuffle(rgb_tuples)
    return rgb_tuples


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
    def __init__(self, viewer):
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
        self.viewer.layers.events.connect(self.on_layers_event)
        viewer.dims.events.order.connect(self.on_order_change)
        viewer.dims.events.ndisplay.connect(lambda _: self.interpolate_button.setEnabled(viewer.dims.ndisplay == 2))
        self.setLayout(layout)
        layout.addStretch()

    def on_layer_selection_change(self, event):
        if event.type == "changed":
            active_layer = self.viewer.layers.selection.active
            if type(active_layer) == Labels:
                self.active_labels_layer = active_layer
                self.dimension_dropdown.setMaximum(self.active_labels_layer.ndim)

    def on_layers_event(self, event):
        if event.type == "inserted":
            print(event.type, event.source[event.index])

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
            start_index = np.argmin(np.arctan2(*(cnt - centroid).T))
            cnt = np.roll(cnt, start_index, 0)
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
                    data[tuple(inter_layer_slice)] = mask
            prev_cnt = cnt
            prev_layer = i
            self.active_labels_layer.refresh()

class AnnotatorModule(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        layout = QVBoxLayout()
        self.fill_objects_checkbox = QCheckBox("autofill objects")
        self.fill_objects_checkbox.setChecked(True)
        self._active_labels_layer = None
        self.viewer = viewer
        self.viewer.layers.selection.events.connect(self.on_layer_selection_change)
        self.fill_objects_checkbox.clicked.connect(self.set_fill_objects)
        layout.addWidget(self.fill_objects_checkbox)
        tabs_widget = QTabWidget()
        annotator_j_widget = QLabel("annotatorJ")
        tabs_widget.addTab(annotator_j_widget, "annotatorJ")
        minimal_surface_widget = QLabel("Minimal surface ")
        tabs_widget.addTab(minimal_surface_widget, "Minimal surface")
        interpolation_widget = InterpolationWidget(viewer)
        tabs_widget.addTab(interpolation_widget, "Interpolation")
        layout.addWidget(tabs_widget)
        self.setLayout(layout)
        self.installEventFilter(self)
        layout.addStretch()
        self.set_fill_objects(True)

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Hide:
            print("sfagasd")
            self.disconnect_all()
            return True
        return super().eventFilter(source, event)

    @property
    def fill_objects(self):
        return self.fill_objects_checkbox.isChecked()

    @property
    def active_labels_layer(self):
        return self._active_labels_layer

    @active_labels_layer.setter
    def active_labels_layer(self, labels_layer):
        if type(labels_layer) is not Labels and labels_layer is not None:
            return
        if self.active_labels_layer is not None and self.fill_holes in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.remove(self.fill_holes)
        if labels_layer is not None:
            labels_layer.mouse_drag_callbacks.append(self.fill_holes)
        self._active_labels_layer = labels_layer
        self.set_fill_objects(self.fill_objects)

    def set_fill_objects(self, state):
        if state and self.active_labels_layer is not None and\
                self.fill_holes not in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.append(self.fill_holes)
        elif not state and self.active_labels_layer is not None and\
                self.fill_holes in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.remove(self.fill_holes)

    def disconnect_all(self):
        self.viewer.layers.selection.events.disconnect(self.on_layer_selection_change)
        if self.active_labels_layer is not None and self.fill_holes in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.remove(self.fill_holes)

    def __del__(self):
        self.disconnect_all()

    def on_layer_selection_change(self, event):
        if event.type == "changed":
            active_layer = self.viewer.layers.selection.active
            if type(active_layer) == Labels:
                self.active_labels_layer = active_layer
            else:
                self.active_labels_layer = None

    def draw_line(self, x1, y1, x2, y2, brush_size, output):
        line_x, line_y = draw.line(x1, y1, x2, y2)
        for x, y in zip(line_x, line_y):
            cx, cy = draw.disk((x, y), math.ceil(brush_size/2+0.1))
            cx = np.clip(cx, 0, output.shape[0] - 1)
            cy = np.clip(cy, 0, output.shape[1] - 1)
            output[cx, cy] = True

    def fill_holes(self, layer: Layer, event):
        if layer.mode != "paint" or layer._ndisplay != 2:
            return
        coordinates = layer.world_to_data(event.position)
        coordinates = tuple(max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
        image_coords = tuple(coordinates[i] for i in range(len(coordinates)) if i in self.viewer.dims.displayed)
        slice_dims = tuple(coordinates[i] if i in self.viewer.dims.not_displayed else slice(None) for i in range(len(coordinates)))
        current_draw = np.zeros_like(layer.data[slice_dims], np.bool)
        start_x, start_y = prev_x, prev_y = image_coords
        cx, cy = draw.disk((start_x, start_y), layer.brush_size/2)
        cx = np.clip(cx, 0, current_draw.shape[0] - 1)
        cy = np.clip(cy, 0, current_draw.shape[1] - 1)
        current_draw[cx, cy] = True
        yield
        while event.type == 'mouse_move':

            coordinates = layer.world_to_data(event.position)
            coordinates = tuple(max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
            image_coords = tuple(coordinates[i] for i in range(len(coordinates)) if i in self.viewer.dims.displayed)
            self.draw_line(prev_x, prev_y, image_coords[-2], image_coords[-1], layer.brush_size, current_draw)
            prev_x, prev_y = image_coords
            yield
        # s = np.asarray([[0, 1, 0],
        #                 [1, 1, 1],
        #                 [0, 1, 0]])
        s = None
        coordinates = layer.world_to_data(event.position)
        coordinates = tuple(
            max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
        image_coords = tuple(coordinates[i] for i in range(len(coordinates)) if i in self.viewer.dims.displayed)
        prev_x, prev_y = image_coords
        self.draw_line(prev_x, prev_y, start_x, start_y, layer.brush_size, current_draw)
        cx, cy = draw.disk((prev_x, prev_y), layer.brush_size/2)
        cx = np.clip(cx, 0, current_draw.shape[0] - 1)
        cy = np.clip(cy, 0, current_draw.shape[1] - 1)
        current_draw[cx, cy] = True
        binary_fill_holes(current_draw, output=current_draw, structure=s)
        layer.data[slice_dims][current_draw] = layer.selected_label
        layer.refresh()
