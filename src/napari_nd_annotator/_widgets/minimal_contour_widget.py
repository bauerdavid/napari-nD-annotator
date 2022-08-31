import warnings

import napari
import skimage.draw
from qtpy.QtWidgets import QSpinBox, QVBoxLayout, QCheckBox, QLabel
from qtpy.QtCore import QMutex

from ._utils.widget_with_layer_list import WidgetWithLayerList
from ..minimal_contour import MinimalContourCalculator
import numpy as np
from napari.layers import Points, Image, Labels
from napari.layers.points._points_constants import Mode
from skimage.filters import gaussian


def bbox_around_points(pts):
    p1 = pts.min(0)
    p2 = pts.max(0)
    size = p2 - p1
    from_i = p1 - size[0]*0.1 - 10
    to_i = p2 + size[0]*0.1 + 10
    return from_i, to_i


class MinimalContourWidget(WidgetWithLayerList):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(viewer, [("image", Image), ("labels", Labels), ("anchor_points", Points)])
        self.viewer = viewer
        self.calculator = MinimalContourCalculator()
        self.anchor_points.combobox.currentIndexChanged.connect(self.set_callbacks)
        self.image.combobox.currentIndexChanged.connect(self.set_image)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Param"))
        self.param_spinbox = QSpinBox()
        self.param_spinbox.setMinimum(1)
        self.param_spinbox.setMaximum(50)
        self.param_spinbox.setValue(5)
        layout.addWidget(self.param_spinbox)

        layout.addWidget(QLabel("Increment label id"))
        self.autoincrease_label_id_checkbox = QCheckBox()
        self.autoincrease_label_id_checkbox.setChecked(True)
        layout.addWidget(self.autoincrease_label_id_checkbox)

        layout.addWidget(QLabel("Point size"))
        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setMinimum(1)
        self.point_size_spinbox.setMaximum(50)
        self.point_size_spinbox.setValue(2)
        layout.addWidget(self.point_size_spinbox)
        layout.addStretch()
        self.setLayout(layout)

        self._img = None
        self.move_mutex = QMutex()
        self.point_triangle = np.zeros((3, 2), dtype=np.float64) - 1  # start point, current position, end point
        self.set_filter_func()

        viewer.layers.events.connect(self.invalidate_filter)

        def change_layer_callback(num):
            def change_layer(_):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if viewer.window.qt_viewer.layers.model().hasIndex(num - 1, 0):
                        index = viewer.window.qt_viewer.layers.model().index(num - 1, 0)
                        layer_name = viewer.window.qt_viewer.layers.model().data(index)
                        viewer.layers.selection.select_only(viewer.layers[layer_name])
            return change_layer
        for i in range(1, 10):
            viewer.bind_key("Control-%d" % i, overwrite=True)(change_layer_callback(i))
        self.from_e_points_layer = viewer.add_points(
            ndim=2,
            name="from E [DO NOT TOUCH] <hidden>",
            size=self.point_size_spinbox.value(),
            face_color="red",
            edge_color="red"
        )
        self.to_s_points_layer = viewer.add_points(
            ndim=2,
            name="to S [DO NOT TOUCH] <hidden>",
            size=self.point_size_spinbox.value(),
            face_color="gray",
            edge_color="gray"
        )
        self.output = viewer.add_points(
            ndim=2,
            name="temp [DO NOT TOUCH] <hidden>",
            size=self.point_size_spinbox.value()
        )
        self.shift_down = False
        self.ctrl_down = False
        self.point_size_spinbox.valueChanged.connect(self.change_point_size)
        self.change_point_size(self.point_size_spinbox.value())
        viewer.dims.events.current_step.connect(self.set_image)

    def showEvent(self, e):
        super().showEvent(e)
        if "Anchors [DO NOT TOUCH]" in self.viewer.layers:
            return
        anchors = self.viewer.add_points(ndim=2, name="Anchors [DO NOT TOUCH]", symbol="x")
        self.anchor_points.layer = anchors
        self.anchor_points.combobox.setEnabled(False)
        self.viewer.layers.events.connect(self.move_temp_to_top)

    def invalidate_filter(self, e):
        # if e.type in ["highlight", "mode", "set_data", "data", "thumbnail", "loaded"]:
        #     return
        self.set_filter_func()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.viewer.window.qt_viewer.layers.model().invalidateFilter()
            # self.set_filter_func()

    def set_filter_func(self):
        def filter_func(row, parent):
            return row < len(self.viewer.layers) and "<hidden>" not in self.viewer.layers[row].name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.viewer.window.qt_viewer.layers.model().filterAcceptsRow = filter_func

    def change_point_size(self, size):
        self.to_s_points_layer.size = size
        self.to_s_points_layer.selected_data = {}
        self.to_s_points_layer.current_size = size

        self.from_e_points_layer.size = size
        self.from_e_points_layer.selected_data = {}
        self.from_e_points_layer.current_size = size

        self.output.size = size
        self.output.selected_data = {}
        self.output.current_size = size
        self.viewer.window.qt_viewer.canvas.native.setFocus()

    def shift_pressed(self, _):
        self.shift_down = True
        self.from_e_points_layer.face_color = "gray"
        self.from_e_points_layer.current_face_color = "gray"
        self.to_s_points_layer.face_color = "red"
        self.to_s_points_layer.current_face_color = "red"
        self.from_e_points_layer.edge_color = "gray"
        self.from_e_points_layer.current_edge_color = "gray"
        self.to_s_points_layer.edge_color = "red"
        self.to_s_points_layer.current_edge_color = "red"
        yield
        self.shift_down = False
        self.from_e_points_layer.face_color = "red"
        self.from_e_points_layer.current_face_color = "red"
        self.to_s_points_layer.face_color = "gray"
        self.to_s_points_layer.current_face_color = "gray"
        self.from_e_points_layer.edge_color = "red"
        self.from_e_points_layer.current_edge_color = "red"
        self.to_s_points_layer.edge_color = "gray"
        self.to_s_points_layer.current_edge_color = "gray"

    def ctrl_pressed(self, _):
        self.ctrl_down = True
        yield
        self.ctrl_down = False

    def ctrl_shift_pressed(self, _):
        shift_callback = self.shift_pressed(_)
        yield from shift_callback
        yield from shift_callback

    def shift_ctrl_pressed(self, _):
        ctrl_callback = self.ctrl_pressed(_)
        yield from ctrl_callback
        yield from ctrl_callback

    def esc_callback(self, _):
        self.clear_all()

    def clear_all(self):
        self.output.data = np.empty((0, 2), dtype=self.output.data.dtype)
        self.anchor_points.layer.data = np.empty((0, 2), dtype=self.output.data.dtype)
        self.from_e_points_layer.data = np.empty((0, 2), dtype=self.output.data.dtype)
        self.to_s_points_layer.data = np.empty((0, 2), dtype=self.output.data.dtype)
        self.point_triangle[:] = -1

    def move_temp_to_top(self, e):
        if e.type in ["highlight", "mode", "set_data", "data", "thumbnail", "loaded"]:
            return
        layer_list = e.source
        with layer_list.events.moved.blocker(), layer_list.events.moving.blocker():
            temp_idx = layer_list.index(self.output)
            if temp_idx != len(layer_list) - 1:
                layer_list.move(temp_idx, -1)
            if self.anchor_points.layer is not None:
                points_idx = layer_list.index(self.anchor_points.layer)
                if points_idx != len(layer_list) - 2:
                    layer_list.move(points_idx, -2)
            to_s_idx = layer_list.index(self.to_s_points_layer)
            if to_s_idx != len(layer_list) - 3:
                layer_list.move(to_s_idx, -3)
            from_e_idx = layer_list.index(self.from_e_points_layer)
            if from_e_idx != len(layer_list) - 4:
                layer_list.move(from_e_idx, -4)

    def estimate(self, image: np.ndarray, param: int):
        from_i, to_i = bbox_around_points(self.point_triangle)
        from_i = np.clip(from_i, 0, np.asarray(image.shape[:2])).astype(int)
        to_i = np.clip(to_i, 0, np.asarray(image.shape[:2])).astype(int)
        results = self.calculator.run(image[from_i[0]:to_i[0], from_i[1]:to_i[1]], self.point_triangle-from_i, param, True, True, True)
        if results is not None:
            for path in results:
                path += from_i
        return results

    def on_mouse_move(self, layer, event):
        if not self.move_mutex.tryLock():
            return
        try:
            if self.image.layer is None or self.anchor_points.layer is None:
                return
            if layer.mode != Mode.ADD:
                return
            self.point_triangle[1] = [event.position[i] for i in range(len(event.position)) if i in event.dims_displayed]
            if np.any(self.point_triangle < 0) or np.any(self.point_triangle >= self._img.shape[:2]):
                return
            if not self.ctrl_down:
                results = self.estimate(self._img, self.param_spinbox.value())
            else:
                results = [
                    np.vstack(skimage.draw.line(
                        int(self.point_triangle[1, 0]),
                        int(self.point_triangle[1, 1]),
                        int(self.point_triangle[0, 0]),
                        int(self.point_triangle[0, 1])

                    )).T,
                    np.vstack(skimage.draw.line(
                        int(self.point_triangle[2, 0]),
                        int(self.point_triangle[2, 1]),
                        int(self.point_triangle[1, 0]),
                        int(self.point_triangle[1, 1])
                    )).T
                ]
            self.from_e_points_layer.data = np.flipud(results[0])
            self.from_e_points_layer.selected_data = {}
            self.to_s_points_layer.data = np.flipud(results[1])
            self.to_s_points_layer.selected_data = {}
        finally:
            self.move_mutex.unlock()

    def set_callbacks(self):
        if self.anchor_points.layer is None:
            return
        self.anchor_points.layer.events.data.connect(self.data_event)
        self.anchor_points.layer.bind_key("Shift", overwrite=True)(self.shift_pressed)
        self.anchor_points.layer.bind_key("Control", overwrite=True)(self.ctrl_pressed)
        self.anchor_points.layer.bind_key("Control-Shift", overwrite=True)(self.ctrl_shift_pressed)
        self.anchor_points.layer.bind_key("Shift-Control", overwrite=True)(self.shift_ctrl_pressed)
        self.anchor_points.layer.bind_key("Escape", overwrite=True)(self.esc_callback)
        self.anchor_points.layer.mouse_double_click_callbacks.append(self.on_double_click)
        if self.on_mouse_move not in self.anchor_points.layer.mouse_move_callbacks:
            self.anchor_points.layer.mouse_move_callbacks.append(self.on_mouse_move)

    def set_image(self):
        image_layer: Image = self.image.layer
        if image_layer is None:
            self._img = None
            return
        if not image_layer.visible:
            image_layer.set_view_slice()
        self.autoincrease_label_id_checkbox.setChecked(image_layer.ndim == 2)
        image = image_layer._data_view.astype(float)
        image = gaussian(image, channel_axis=3 if image.ndim == 3 else None)
        if image.ndim == 2:
            image = np.concatenate([image[..., np.newaxis]] * 3, axis=2)
        image = (image - image.min()) / (image.max() - image.min())
        self._img = image

    def data_event(self, event):
        if event.source != self.anchor_points.layer or len(self.anchor_points.layer.data) == 0:
            return
        anchor_data = self.anchor_points.layer.data
        anchor_data[-1] = np.clip(anchor_data[-1], 0, self._img.shape[:2])
        if len(anchor_data) > 1 and np.all(np.round(anchor_data[-1]) == np.round(anchor_data[-2])):
            with self.anchor_points.layer.events.data.blocker():
                self.anchor_points.layer.data = self.anchor_points.layer.data[:-1]
            self.anchor_points.layer.refresh()
            return
        self.anchor_points.layer.refresh()
        if self.shift_down:
            with self.anchor_points.layer.events.data.blocker():
                self.anchor_points.layer.data = np.roll(self.anchor_points.layer.data, 1, 0)
            if len(self.to_s_points_layer.data):
                self.output.data = np.concatenate([self.to_s_points_layer.data, self.output.data], 0)
        else:
            if len(self.from_e_points_layer.data):
                self.output.data = np.concatenate([self.output.data, self.from_e_points_layer.data], 0)
        self.point_triangle[-1] = self.anchor_points.layer.data[0]
        self.point_triangle[0] = self.anchor_points.layer.data[-1]

    def on_double_click(self, *args):
        if self.shift_down and len(self.from_e_points_layer.data):
            self.output.data = np.concatenate([self.output.data, self.from_e_points_layer.data], 0)
        elif not self.shift_down and len(self.to_s_points_layer.data):
                self.output.data = np.concatenate([self.to_s_points_layer.data, self.output.data], 0)
        self.points_to_mask()
        self.clear_all()
        if self.labels.layer and self.autoincrease_label_id_checkbox.isChecked():
            self.labels.layer.selected_label += 1

    def points_to_mask(self):
        if self._img is None or len(self.output.data) == 0:
            return
        elif self.labels.layer is None:
            warnings.warn("Missing output labels layer.")
            return
        if not self.labels.layer.visible:
            self.labels.layer.set_view_slice()
        contour = self.output.data
        image_shape = self._img.shape[:2]
        mask = skimage.draw.polygon2mask(image_shape, contour)
        self.labels.layer._slice.image.raw[mask] = self.labels.layer.selected_label
        self.labels.layer.refresh()
