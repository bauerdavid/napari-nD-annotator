from qtpy.QtCore import QObject, QEvent, Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QTabWidget, QPushButton
from napari import Viewer
from napari.layers import Labels, Layer
import numpy as np
from skimage.draw import draw
from scipy.ndimage import binary_fill_holes
import math

from .interpolation_widget import InterpolationWidget
from .minimal_contour_widget import MinimalContourWidget
from ._utils.callbacks import (
    extend_mask,
    reduce_mask,
    increase_brush_size,
    decrease_brush_size,
    scroll_to_next,
    scroll_to_prev,
    increment_selected_label,
    decrement_selected_label
)
from ._utils.help_dialog import HelpDialog
def check_connectivity(event):
    layer = event.source
    current_slice = layer._slice.image.raw


class AnnotatorWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.history_idx = 0
        layout = QVBoxLayout()
        self.fill_objects_checkbox = QCheckBox("autofill objects")
        self.fill_objects_checkbox.setChecked(True)
        self.fill_objects_checkbox.setToolTip("When drawing labels,"
                                              " close the drawn curve and fill its area after releasing the mouse")
        self._active_labels_layer = None
        self.drawn_region_history = dict()
        self.drawn_slice_history = dict()
        self.values_history = dict()
        self.viewer = viewer
        self.viewer.layers.selection.events.connect(self.on_layer_selection_change)
        self.fill_objects_checkbox.clicked.connect(self.set_fill_objects)
        layout.addWidget(self.fill_objects_checkbox)

        tabs_widget = QTabWidget()
        self.interpolation_widget = InterpolationWidget(viewer)
        tabs_widget.addTab(self.interpolation_widget, "Interpolation")

        self.minimal_contour_widget = MinimalContourWidget(viewer)
        tabs_widget.addTab(self.minimal_contour_widget, "Minimal Contour")

        layout.addWidget(tabs_widget)

        help_layout = QVBoxLayout()
        help_layout.setAlignment(Qt.AlignRight)
        help_button = QPushButton("?")
        help_button.setToolTip("Help")
        help_button.clicked.connect(self.show_help_window)
        help_button.setFixedSize(20, 20)
        help_layout.addWidget(help_button)
        layout.addLayout(help_layout)
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
        self.set_labels_callbacks()

    def unset_labels_callbacks(self):
        if self.active_labels_layer is None:
            return
        if check_connectivity in self.active_labels_layer.events.data.callbacks:
            self.active_labels_layer.events.data.disconnect(check_connectivity)
        self.active_labels_layer.bind_key("Control-+", overwrite=True)(None)
        self.active_labels_layer.bind_key("Control--", overwrite=True)(None)
        self.active_labels_layer.bind_key("Q", overwrite=True)(None)
        self.active_labels_layer.bind_key("E", overwrite=True)(None)
        self.active_labels_layer.bind_key("A", overwrite=True)(None)
        self.active_labels_layer.bind_key("D", overwrite=True)(None)

    def set_labels_callbacks(self):
        if self.active_labels_layer is None:
            return
        if check_connectivity not in self.active_labels_layer.events.data.callbacks:
            self.active_labels_layer.events.data.connect(check_connectivity)
        self.active_labels_layer.bind_key("Control-Z", overwrite=True)(self.undo)
        self.active_labels_layer.bind_key("Control-+", overwrite=True)(extend_mask)
        self.active_labels_layer.bind_key("Control--", overwrite=True)(reduce_mask)
        self.active_labels_layer.bind_key("Q", overwrite=True)(decrement_selected_label)
        self.active_labels_layer.bind_key("E", overwrite=True)(increment_selected_label)
        self.active_labels_layer.bind_key("A", overwrite=True)(scroll_to_prev(self.viewer))
        self.active_labels_layer.bind_key("D", overwrite=True)(scroll_to_next(self.viewer))
        self.active_labels_layer.bind_key("W", overwrite=True)(increase_brush_size)
        self.active_labels_layer.bind_key("S", overwrite=True)(decrease_brush_size)

    def set_fill_objects(self, state):
        if self.active_labels_layer is None:
            return
        if state and self.isVisible():
            if self.fill_holes not in self.active_labels_layer.mouse_drag_callbacks:
                self.active_labels_layer.mouse_drag_callbacks.append(self.fill_holes)
                self.active_labels_layer.mouse_drag_callbacks.append(self.incr_history_idx)
        elif not state:
            if self.fill_holes in self.active_labels_layer.mouse_drag_callbacks:
                self.active_labels_layer.mouse_drag_callbacks.remove(self.fill_holes)
                self.active_labels_layer.mouse_drag_callbacks.remove(self.incr_history_idx)

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

    @staticmethod
    def draw_line(x1, y1, x2, y2, brush_size, output):
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
        image_coords = tuple(coordinates[i] for i in range(len(coordinates)) if i in event.dims_displayed)
        slice_dims = tuple(coordinates[i] if i not in event.dims_displayed else slice(None) for i in range(len(coordinates)))
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
            image_coords = tuple(coordinates[i] for i in range(len(coordinates)) if i in event.dims_displayed)
            AnnotatorWidget.draw_line(prev_x, prev_y, image_coords[-2], image_coords[-1], layer.brush_size, current_draw)
            prev_x, prev_y = image_coords
            yield
        # s = np.asarray([[0, 1, 0],
        #                 [1, 1, 1],
        #                 [0, 1, 0]])
        s = None
        coordinates = layer.world_to_data(event.position)
        coordinates = tuple(
            max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
        image_coords = tuple(coordinates[i] for i in range(len(coordinates)) if i in event.dims_displayed)
        prev_x, prev_y = image_coords
        AnnotatorWidget.draw_line(prev_x, prev_y, start_x, start_y, layer.brush_size, current_draw)
        cx, cy = draw.disk((prev_x, prev_y), layer.brush_size/2)
        cx = np.clip(cx, 0, current_draw.shape[0] - 1)
        cy = np.clip(cy, 0, current_draw.shape[1] - 1)
        current_draw[cx, cy] = True
        binary_fill_holes(current_draw, output=current_draw, structure=s)
        if layer.preserve_labels:
            current_draw = current_draw & (layer.data[slice_dims] == 0)
        self.drawn_region_history[self.history_idx] = current_draw
        self.drawn_slice_history[self.history_idx] = slice_dims
        self.values_history[self.history_idx] = layer.data[slice_dims][current_draw]
        layer.data[slice_dims][current_draw] = layer.selected_label
        layer.events.data()
        layer.refresh()

    def undo(self, layer):
        last_drawn_region = self.drawn_region_history.get(self.history_idx, None)
        last_drawn_slice = self.drawn_slice_history.get(self.history_idx, None)
        prev_values = self.values_history.get(self.history_idx, None)
        if self.fill_objects_checkbox.isChecked() and last_drawn_region is not None:
            layer.data[last_drawn_slice][last_drawn_region] = prev_values
            del self.drawn_region_history[self.history_idx]
            del self.drawn_slice_history[self.history_idx]
            del self.values_history[self.history_idx]
        self.decr_history_idx()
        layer.undo()

    def incr_history_idx(self, *args):
        self.history_idx += 1

    def decr_history_idx(self, *args):
        if self.history_idx >= 0:
            self.history_idx -= 1

    def show_help_window(self):
        dialog = HelpDialog(self)
        dialog.show()