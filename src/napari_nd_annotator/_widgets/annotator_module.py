from qtpy.QtCore import QObject, QEvent
from qtpy.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QTabWidget, QLabel
from napari import Viewer
from napari.layers import Labels, Layer
import numpy as np
from skimage.draw import draw
from scipy.ndimage import binary_fill_holes
import math

from .interpolation_widget import InterpolationWidget


class AnnotatorWidget(QWidget):
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
        interpolation_widget = InterpolationWidget(viewer)
        tabs_widget.addTab(interpolation_widget, "Interpolation")
        layout.addWidget(tabs_widget)
        self.setLayout(layout)
        self.installEventFilter(self)
        layout.addStretch()
        self.set_fill_objects(True)
        self.on_layer_selection_change()

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Hide:
            self.disconnect_all()
            return True
        elif event.type() == QEvent.Show:
            self.connect_all()
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
        if not isinstance(labels_layer, Labels) and labels_layer is not None:
            return
        self._active_labels_layer = labels_layer
        self.set_fill_objects(self.fill_objects)

    def set_fill_objects(self, state):
        if state and self.isVisible() and self.active_labels_layer is not None and\
                self.fill_holes not in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.append(self.fill_holes)
        elif not state and self.active_labels_layer is not None and\
                self.fill_holes in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.remove(self.fill_holes)

    def disconnect_all(self):
        if self.active_labels_layer is not None and self.fill_holes in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.remove(self.fill_holes)

    def connect_all(self):
        if self.fill_objects and self.active_labels_layer is not None and self.fill_holes not in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.append(self.fill_holes)

    def on_layer_selection_change(self, event=None):
        if event is None or event.type == "changed":
            active_layer = self.viewer.layers.selection.active
            if isinstance(active_layer, Labels):
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
