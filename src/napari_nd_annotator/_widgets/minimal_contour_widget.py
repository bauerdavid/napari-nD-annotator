import warnings
import napari
import skimage.draw
from napari._qt.layer_controls.qt_image_controls_base import _QDoubleRangeSlider
from napari.layers.labels._labels_utils import get_dtype
from napari.utils._dtype import get_dtype_limits
from skimage.morphology import binary_dilation, binary_erosion
from napari.utils.events import Event
from qtpy.QtWidgets import (
    QWidget,
    QSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QComboBox,
    QPushButton,
    QSizePolicy,
    QMessageBox
)
from qtpy.QtCore import QMutex, QThread, QObject, Signal, Qt, QEvent
from qtpy.QtGui import QCursor, QPixmap, QImage
from superqt import QLargeIntSpinBox
from ._utils import(
    QDoubleSlider,
    ProgressWidget,
    WidgetWithLayerList,
    CollapsibleWidget,
    CollapsibleWidgetGroup,
    ImageProcessingWidget
)
from ._utils.changeable_color_box import QtChangeableColorBox
from ._utils.callbacks import (
    extend_mask,
    reduce_mask,
    increase_brush_size,
    decrease_brush_size,
    scroll_to_next,
    scroll_to_prev,
    increment_selected_label,
    decrement_selected_label,
    LOCK_CHAR
)
from .._helper_functions import layer_dims_displayed, layer_dims_not_displayed
from ..minimal_contour import MinimalContourCalculator, FeatureManager
import numpy as np
from napari.layers import Image, Labels
from napari.layers.points._points_constants import Mode
from skimage.filters import gaussian
import scipy.fft
import colorsys

GRADIENT_BASED = 0
INTENSITY_BASED = 2

ASSYM_GRADIENT_TEXT = "Gradient [assymetric]"
SYM_GRADIENT_TEXT = "Gradient [symmetric]"
HIGH_INTENSITY_TEXT = "High intensity"
LOW_INTENSITY_TEXT = "Low intensity"
CUSTOM_TEXT = "Custom"
FEATURE_TEXTS = [ASSYM_GRADIENT_TEXT, SYM_GRADIENT_TEXT, HIGH_INTENSITY_TEXT, LOW_INTENSITY_TEXT, CUSTOM_TEXT]
FEATURE_TYPES = [GRADIENT_BASED, INTENSITY_BASED, INTENSITY_BASED, INTENSITY_BASED, INTENSITY_BASED]

DEMO_SIZE = 200


def bbox_around_points(pts):
    p1 = pts.min(0)
    p2 = pts.max(0)
    size = p2 - p1
    from_i = p1 - size[0]*0.1 - 10
    to_i = p2 + size[0]*0.1 + 10
    return from_i, to_i


class MinimalContourWidget(WidgetWithLayerList):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(viewer, [("image", Image), ("labels", Labels)], "nd_annotator_mc", parent=parent)
        self.viewer = viewer
        self.calculator = MinimalContourCalculator(None, 3)
        self.progress_dialog = ProgressWidget(self, message="Drawing mask...")
        self.move_mutex = QMutex()
        self.draw_worker = self.DrawWorker()
        self.draw_thread = QThread()
        self.draw_worker.moveToThread(self.draw_thread)
        self.draw_worker.done.connect(self.draw_thread.quit)
        self.draw_worker.done.connect(self.set_mask)
        self.draw_worker.done.connect(lambda: self.progress_dialog.setVisible(False))
        self.draw_thread.started.connect(self.draw_worker.run)
        self._img = None
        self.apply_contrast_limits = True
        self._prev_img_layer = self.image.layer
        self._orig_image = self.image.layer.data if self.image.layer is not None else None
        self.point_triangle = np.zeros((3, 2), dtype=np.float64) - 1  # start point, current position, end point
        self.remove_last_anchor = False
        self.last_added_with_shift = None
        self.last_segment_length = None
        self.prev_n_anchor_points = 0
        self.prev_labels_layer = self.labels.layer
        self.feature_inverted = False
        self.prev_timer_id = None
        self.feature_manager = FeatureManager(viewer)

        # Ctrl+Z handling
        self.change_idx = dict()
        self.prev_vals = dict()

        layout = QVBoxLayout()
        features_widget = CollapsibleWidget("Image features", parent=self)
        features_layout = QVBoxLayout()
        features_layout.addWidget(QLabel("Used feature", parent=self))
        self.feature_editor = ImageProcessingWidget(self._img, viewer, parent=self)
        self.feature_dropdown = QComboBox(parent=self)
        for text, value in zip(FEATURE_TEXTS, FEATURE_TYPES):
            self.feature_dropdown.addItem(text, value)
        features_layout.addWidget(self.feature_dropdown)
        self.add_stored_widget("feature_dropdown")
        self.feature_editor.setVisible(self.feature_dropdown.currentText() == CUSTOM_TEXT)
        features_layout.addWidget(self.feature_editor)

        features_layout.addWidget(QLabel("Param", parent=self))
        self.param_spinbox = QSpinBox(parent=self)
        self.param_spinbox.setMinimum(1)
        self.param_spinbox.setMaximum(50)
        self.param_spinbox.setToolTip("Increasing this parameter will cause contours to be more aligned\n"
                                      "with the selected image feature (with a lower number the contour will be\n"
                                      "closer to the Euclidean shortest path")
        self.calculator.set_param(self.param_spinbox.value())
        features_layout.addWidget(self.param_spinbox)
        self.add_stored_widget("param_spinbox")
        local_correction_checkbox = QCheckBox("Local intensity correction", parent=self)
        local_correction_checkbox.clicked.connect(lambda is_checked: self.calculator.set_use_local_maximum(is_checked))
        features_layout.addWidget(local_correction_checkbox)
        features_widget.setLayout(features_layout)
        layout.addWidget(features_widget, 0, Qt.AlignTop)

        blur_widget = CollapsibleWidget("Blurring", parent=self)
        blur_layout = QVBoxLayout()
        self.blur_image_checkbox = QCheckBox("Blur image")
        self.blur_image_checkbox.setToolTip("When checked, the image will be blurred\n"
                                            "before calculating minimal contour")
        blur_layout.addWidget(self.blur_image_checkbox)
        self.add_stored_widget("blur_image_checkbox")
        blur_image_widget = QWidget(parent=self)
        blur_image_layout = QVBoxLayout()
        blur_image_layout.setContentsMargins(0, 0, 0, 0)
        blur_image_layout.addWidget(QLabel("Blur sigma"))
        self.blur_image_slider = QDoubleSlider(parent=self)
        self.blur_image_slider.setMaximum(20)
        self.blur_image_slider.setOrientation(Qt.Horizontal)
        blur_image_layout.addWidget(self.blur_image_slider)
        self.add_stored_widget("blur_image_slider")
        self.blur_image_slider.valueChanged.connect(self.update_demo_image)
        demo_widget = CollapsibleWidget("Example", parent=self)
        self.demo_image = QLabel(parent=self)
        self.update_demo_image()
        demo_layout = QVBoxLayout()
        demo_layout.addWidget(self.demo_image)
        demo_widget.setLayout(demo_layout)
        blur_image_layout.addWidget(demo_widget)

        self.apply_blurring_button = QPushButton("Apply blurring", parent=self)
        self.apply_blurring_button.setEnabled(self.blur_image_checkbox.isChecked())
        self.apply_blurring_button.clicked.connect(self.apply_blurring)
        blur_image_layout.addWidget(self.apply_blurring_button)
        blur_image_widget.setLayout(blur_image_layout)
        self.blur_image_widget = blur_image_widget
        blur_layout.addWidget(blur_image_widget)
        blur_widget.setLayout(blur_layout)
        layout.addWidget(blur_widget, 0, Qt.AlignTop)

        contour_widget = CollapsibleWidget("Contour", parent=self)
        contour_layout = QVBoxLayout()
        smooth_contour_layout = QHBoxLayout()
        self.smooth_contour_checkbox = QCheckBox("Smooth contour", parent=self)
        self.smooth_contour_checkbox.setToolTip("When checked, the finished contour will be smoothed\n"
                                                "to remove minor noise using Fourier transformation.")
        self.smooth_contour_spinbox = QDoubleSlider(Qt.Horizontal, self)
        self.smooth_contour_spinbox.setMinimum(0.)
        self.smooth_contour_spinbox.setMaximum(1.)
        # self.smooth_contour_spinbox.setTickInterval(0.05)
        self.smooth_contour_spinbox.setToolTip("Number of Fourier coefficients to approximate the contour.\n"
                                               "Lower number -> smoother contour\n"
                                               "Higher number -> more faithful to the original")
        self.add_stored_widget("smooth_contour_checkbox")
        self.smooth_contour_spinbox.setVisible(self.smooth_contour_checkbox.isChecked())
        smooth_contour_layout.addWidget(self.smooth_contour_checkbox)
        smooth_contour_layout.addWidget(self.smooth_contour_spinbox)
        self.add_stored_widget("smooth_contour_spinbox")
        contour_layout.addLayout(smooth_contour_layout)

        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("Point size", parent=self))
        self.point_size_spinbox = QSpinBox(parent=self)
        self.point_size_spinbox.setMinimum(1)
        self.point_size_spinbox.setMaximum(50)
        self.point_size_spinbox.setToolTip("Point size for contour display.")
        point_size_layout.addWidget(self.point_size_spinbox)
        self.add_stored_widget("point_size_spinbox")
        contour_layout.addLayout(point_size_layout)
        contour_widget.setLayout(contour_layout)
        layout.addWidget(contour_widget, 0, Qt.AlignTop)

        label_widget = CollapsibleWidget("Label options", parent=self)
        label_layout = QVBoxLayout()

        self.selectionSpinBox = QLargeIntSpinBox(parent=self)
        if self.labels.layer is not None:
            dtype_lims = get_dtype_limits(get_dtype(self.labels.layer))
        else:
            dtype_lims = 0, np.iinfo(int).max
        self.selectionSpinBox.setRange(*dtype_lims)
        self.selectionSpinBox.setKeyboardTracking(False)
        self.selectionSpinBox.setAlignment(Qt.AlignCenter)
        self.selectionSpinBox.setToolTip("Selected label for the current 'labels' layer.")
        self.selected_id_label = QLabel(parent=self)
        self.selected_id_label.setFixedSize(20, 20)
        self.selected_id_label.setAlignment(Qt.AlignCenter)
        self.selected_id_label.setSizePolicy(QSizePolicy.Fixed, self.selected_id_label.sizePolicy().verticalPolicy())

        self.selected_id_label.setWindowFlags(Qt.Window
                                              | Qt.FramelessWindowHint
                                              | Qt.WindowStaysOnTopHint)
        self.selected_id_label.setAttribute(Qt.WA_ShowWithoutActivating)
        self.selected_id_label.installEventFilter(self)
        color_layout = QHBoxLayout()
        self.colorBox = QtChangeableColorBox(self.labels.layer)
        color_layout.addWidget(self.colorBox)
        color_layout.addWidget(self.selectionSpinBox)
        self.on_label_change()
        self._on_selected_label_change()
        label_layout.addLayout(color_layout)

        self.autoincrease_label_id_checkbox = QCheckBox("Increment label id", parent=self)
        self.autoincrease_label_id_checkbox.setChecked(True)
        self.autoincrease_label_id_checkbox.setToolTip("When checked, selected label will be incremented\n"
                                                       "after completing the contour")
        label_layout.addWidget(self.autoincrease_label_id_checkbox)

        label_manipulation_layout = QHBoxLayout()

        reduce_mask_button = QPushButton("Reduce label", parent=self)
        reduce_mask_button.clicked.connect(lambda: reduce_mask(self.labels.layer))
        reduce_mask_button.setToolTip("Reduce the selected label by one pixel around the borders\n"
                                      "using erosion")
        label_manipulation_layout.addWidget(reduce_mask_button)

        extend_mask_button = QPushButton("Extend label", parent=self)
        extend_mask_button.clicked.connect(lambda: extend_mask(self.labels.layer))
        extend_mask_button.setToolTip("Extend the selected label by one pixel around the borders\n"
                                      "using dilation")
        label_manipulation_layout.addWidget(extend_mask_button)
        label_layout.addLayout(label_manipulation_layout)
        label_widget.setLayout(label_layout)
        layout.addWidget(label_widget, 0, Qt.AlignTop)
        self._collapsible_group = CollapsibleWidgetGroup([features_widget, blur_widget, contour_widget, label_widget])

        layout.addStretch()
        self.setLayout(layout)

        def change_layer_callback(num):
            def change_layer(_):
                if num-1 < len(viewer.layers):
                    viewer.layers.selection.select_only(viewer.layers[num-1])
                else:
                    warnings.warn("%d is out of range (number of layers: %d)" % (num, len(viewer.layers)))
            return change_layer
        for i in range(1, 10):
            viewer.bind_key("Control-%d" % i, overwrite=True)(change_layer_callback(i))
        self.from_e_points_layer = viewer.add_points(
            ndim=3,
            name="%s from E" % LOCK_CHAR,
            size=self.point_size_spinbox.value(),
            face_color="red",
            edge_color="red"
        )
        self.from_e_points_layer.editable = False
        self.to_s_points_layer = viewer.add_points(
            ndim=3,
            name="%s to S" % LOCK_CHAR,
            size=self.point_size_spinbox.value(),
            face_color="gray",
            edge_color="gray"
        )
        self.to_s_points_layer.editable = False
        self.output = viewer.add_points(
            ndim=3,
            name="%s temp" % LOCK_CHAR,
            size=self.point_size_spinbox.value(),
        )
        self.output.editable = False
        self.anchor_points = self.viewer.add_points(ndim=3, name="Anchors [DO NOT ALTER]", symbol="x")
        self.anchor_points.mode = "add"
        self.modifiers = None
        self.point_size_spinbox.valueChanged.connect(self.change_point_size)
        self.image.combobox.currentIndexChanged.connect(self.store_orig_image)
        self.image.combobox.currentIndexChanged.connect(self.apply_blurring)
        self.image.combobox.currentIndexChanged.connect(self.set_image)
        self.image.combobox.currentIndexChanged.connect(self.update_demo_image)
        self.feature_dropdown.currentIndexChanged.connect(self.on_feature_change)
        self.feature_editor.script_worker.done.connect(self.set_features)
        self.param_spinbox.valueChanged.connect(self.on_param_change)
        self.blur_image_checkbox.stateChanged.connect(self.set_use_smoothing)
        self.smooth_contour_checkbox.stateChanged.connect(
            lambda checked: self.smooth_contour_spinbox.setVisible(checked))
        self.selectionSpinBox.valueChanged.connect(self.change_selected_label)
        self.labels.combobox.currentIndexChanged.connect(self.on_label_change)
        viewer.dims.events.current_step.connect(self.delayed_set_image, position="last")
        viewer.dims.events.ndisplay.connect(self.set_image, position="last")
        viewer.dims.events.order.connect(self.set_image, position="last")
        viewer.dims.events.current_step.connect(self.update_demo_image, position="last")
        self.viewer.layers.events.inserted.connect(self.move_temp_to_top, position="last")
        self.viewer.layers.events.moved.connect(self.move_temp_to_top, position="last")
        self.viewer.layers.events.removed.connect(lambda e: self.feature_manager.remove_features(e.value))
        self.viewer.bind_key("Control-Tab", overwrite=True)(self.swap_selection)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.viewer.window.qt_viewer.window().installEventFilter(self)
            self.viewer.window.qt_viewer.window().setMouseTracking(True)
        self.change_point_size(self.point_size_spinbox.value())
        self.store_orig_image()
        self.set_use_smoothing(self.blur_image_checkbox.isChecked())
        self.smooth_contour_spinbox.setVisible(self.smooth_contour_checkbox.isChecked())
        self.apply_blurring()
        self.set_image()
        self.update_demo_image()
        self.on_feature_change()
        self.on_param_change(self.param_spinbox.value())
        self.set_callbacks()
        self.move_temp_to_top()
        self.update_label_tooltip()

    def set_use_smoothing(self, use_smoothing):
        self.blur_image_widget.setVisible(use_smoothing)
        self.apply_blurring_button.setEnabled(use_smoothing)
        self.apply_blurring()
        self.set_image()

    def set_features(self, data):
        if data is None:
            return
        if data.ndim == 2:
            data = np.concatenate([data[..., np.newaxis]] * 3, -1)
        self.calculator.set_image(data, np.empty((0, 0,)), np.empty((0, 0,)))

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.viewer.window.qt_viewer.canvas.native.setFocus()

    def shift_pressed(self, _):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.labels.layer is not None and len(self.anchor_points.data) == 0 and self.viewer.window.qt_viewer.canvas.native.hasFocus():
                self.selected_id_label.show()
        self.from_e_points_layer.face_color = "gray"
        self.from_e_points_layer.current_face_color = "gray"
        self.to_s_points_layer.face_color = "red"
        self.to_s_points_layer.current_face_color = "red"
        self.from_e_points_layer.edge_color = "gray"
        self.from_e_points_layer.current_edge_color = "gray"
        self.to_s_points_layer.edge_color = "red"
        self.to_s_points_layer.current_edge_color = "red"
        yield
        self.selected_id_label.hide()
        self.from_e_points_layer.face_color = "red"
        self.from_e_points_layer.current_face_color = "red"
        self.to_s_points_layer.face_color = "gray"
        self.to_s_points_layer.current_face_color = "gray"
        self.from_e_points_layer.edge_color = "red"
        self.from_e_points_layer.current_edge_color = "red"
        self.to_s_points_layer.edge_color = "gray"
        self.to_s_points_layer.current_edge_color = "gray"

    @property
    def ctrl_down(self):
        return self.modifiers & Qt.ControlModifier if self.modifiers is not None else False

    @property
    def shift_down(self):
        return self.modifiers & Qt.ShiftModifier if self.modifiers is not None else False

    @property
    def alt_down(self):
        return self.modifiers & Qt.AltModifier if self.modifiers is not None else False

    def ctrl_shift_pressed(self, _):
        shift_callback = self.shift_pressed(_)
        yield from shift_callback
        yield from shift_callback

    def ctrl_z_callback(self, _):
        if self.labels.layer in self.change_idx:
            idx = self.change_idx[self.labels.layer]
            vals = self.prev_vals[self.labels.layer]
            self.labels.layer.data[idx] = vals
            self.labels.layer.events.data()
            self.labels.layer.refresh()
            del self.change_idx[self.labels.layer]
            del self.prev_vals[self.labels.layer]
        else:
            warnings.warn("There's nothing to revert.")

    def esc_callback(self, _):
        self.clear_all()

    def clear_all(self):
        self.output.data = np.empty((0, 3), dtype=self.output.data.dtype)
        with self.anchor_points.events.data.blocker():
            self.anchor_points.data = np.empty((0, 3), dtype=self.output.data.dtype)
        self.prev_n_anchor_points = 0
        self.from_e_points_layer.data = np.empty((0, 3), dtype=self.output.data.dtype)
        self.to_s_points_layer.data = np.empty((0, 3), dtype=self.output.data.dtype)
        self.point_triangle[:] = -1
        self.prev_n_anchor_points = 0

    @property
    def image_data(self):
        return self._img

    @property
    def features(self):
        return self.feature_editor.features

    def on_label_change(self, _=None):
        if self.labels.layer is not None:
            dtype_lims = get_dtype_limits(get_dtype(self.labels.layer))
            self.selectionSpinBox.setValue(self.labels.layer.selected_label)
        else:
            dtype_lims = 0, np.iinfo(int).max
        self.selectionSpinBox.setRange(*dtype_lims)
        self.colorBox.layer = self.labels.layer
        if self.prev_labels_layer is not None:
            self.prev_labels_layer.events.selected_label.disconnect(self._on_selected_label_change)
            if self.on_mouse_wheel in self.prev_labels_layer.mouse_wheel_callbacks:
                self.prev_labels_layer.mouse_wheel_callbacks.remove(self.on_mouse_wheel)
            self.prev_labels_layer.bind_key("Shift", overwrite=True)(None)
            self.prev_labels_layer.bind_key("Alt", overwrite=True)(None)
            self.prev_labels_layer.bind_key("Control-+", overwrite=True)(None)
            self.prev_labels_layer.bind_key("Control--", overwrite=True)(None)
        if self.labels.layer is not None:
            self.labels.layer.events.selected_label.connect(
                self._on_selected_label_change
            )
            if self.on_mouse_wheel not in self.labels.layer.mouse_wheel_callbacks:
                self.labels.layer.mouse_wheel_callbacks.append(self.on_mouse_wheel)
            self.labels.layer.bind_key("Shift", overwrite=True)(self.shift_pressed)
            self._on_selected_label_change()
        self.prev_labels_layer = self.labels.layer

    def _on_selected_label_change(self, *args):
        """Receive layer model label selection change event and update spinbox."""
        if self.labels.layer is None:
            return
        with self.labels.layer.events.selected_label.blocker():
            value = self.labels.layer.selected_label
            self.selectionSpinBox.setValue(value)
            self.update_label_tooltip()

    def change_selected_label(self, value):
        self.labels.layer.selected_label = value
        self.selectionSpinBox.clearFocus()
        self.setFocus()

    def move_temp_to_top(self, e=None):
        layer_list = self.viewer.layers
        try:
            with layer_list.events.moved.blocker(self.move_temp_to_top):
                temp_idx = layer_list.index(self.output)
                if temp_idx != len(layer_list) - 1:
                    layer_list.move(temp_idx, -1)
                if self.anchor_points is not None:
                    points_idx = layer_list.index(self.anchor_points)
                    if points_idx != len(layer_list) - 2:
                        layer_list.move(points_idx, -2)
                if self.to_s_points_layer in layer_list:
                    to_s_idx = layer_list.index(self.to_s_points_layer)
                    if to_s_idx != len(layer_list) - 3:
                        layer_list.move(to_s_idx, -3)
                if self.from_e_points_layer in layer_list:
                    from_e_idx = layer_list.index(self.from_e_points_layer)
                    if from_e_idx != len(layer_list) - 4:
                        layer_list.move(from_e_idx, -4)
        except KeyError:
            ...

    def estimate(self, image: np.ndarray):
        from_i, to_i = bbox_around_points(self.point_triangle)
        from_i = np.clip(from_i, 0, np.asarray(image.shape[:2])).astype(int)
        to_i = np.clip(to_i, 0, np.asarray(image.shape[:2])).astype(int)
        self.calculator.set_boundaries(from_i[1], from_i[0], to_i[1], to_i[0])
        results = self.calculator.run(
            self.point_triangle,
            True,
            True,
            True
        )
        return results

    def on_feature_change(self, _=None):
        current_text = self.feature_dropdown.currentText()
        self.feature_editor.setVisible(current_text == CUSTOM_TEXT)
        self.feature_inverted = current_text == LOW_INTENSITY_TEXT
        self.set_image()
        self.calculator.set_method(self.feature_dropdown.currentData())

    def on_mouse_move(self, layer, event):
        if not self.move_mutex.tryLock():
            return
        try:
            if self.image.layer is None or self.anchor_points is None:
                return
            if layer.mode != Mode.ADD:
                return
            self.point_triangle[1] = np.asarray(event.position)[[i for i in layer_dims_displayed(self.anchor_points)]]
            displayed_shape = np.asarray(self.image_data.shape)[:2]
            if np.any(self.point_triangle[1] < 0) or np.any(self.point_triangle[1] >= displayed_shape):
                self.point_triangle[1] = np.clip(self.point_triangle[1], 0, np.subtract(displayed_shape, 1))
            if np.any(self.point_triangle < 0) or np.any(self.point_triangle >= displayed_shape):
                return
            if not self.ctrl_down:
                if self.feature_editor.isVisible():
                    if self.features is None:
                        warnings.warn("Feature image not calculated. Run script using the 'Set' button.")
                        return
                    results = self.estimate(self.features)
                else:
                    if self.image_data is None:
                        warnings.warn("Image was not set.")
                        return
                    results = self.estimate(self.image_data)
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
            new_e_data = np.tile(np.where([i in layer_dims_displayed(self.anchor_points) for i in range(3)], np.nan, self.anchor_points.data[0]), [len(results[0]), 1])
            new_s_data = np.tile(np.where([i in layer_dims_displayed(self.anchor_points) for i in range(3)], np.nan, self.anchor_points.data[0]), [len(results[1]), 1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_e_data[np.isnan(new_e_data)] = np.flipud(results[0][:, self.image.layer._get_order()[:2]]).reshape(-1)
                new_s_data[np.isnan(new_s_data)] = np.flipud(results[1][:, self.image.layer._get_order()[:2]]).reshape(-1)
            
            self.from_e_points_layer.data = new_e_data
            self.to_s_points_layer.data = new_s_data
            self.from_e_points_layer.selected_data = {}
            self.to_s_points_layer.selected_data = {}
        finally:
            self.move_mutex.unlock()

    def on_mouse_wheel(self, _, event):
        if self.labels.layer is None:
            return
        delta = (np.sign(event.delta)).astype(int)
        if self.shift_down:
            self.labels.layer.selected_label = max(0, self.labels.layer.selected_label + delta[1])
        elif self.alt_down:
            diff = delta[0]
            diff *= min(max(1, self.labels.layer.brush_size//10), 5)
            self.labels.layer.brush_size = max(1, self.labels.layer.brush_size + diff)

    def set_callbacks(self):
        if self.anchor_points is None:
            return
        self.anchor_points.events.data.connect(self.data_event)
        self.anchor_points.events.mode.connect(self.add_mode_only)
        self.anchor_points.bind_key("Shift", overwrite=True)(self.shift_pressed)
        self.anchor_points.bind_key("Control-Shift", overwrite=True)(self.ctrl_shift_pressed)
        self.anchor_points.bind_key("Escape", overwrite=True)(self.esc_callback)
        self.anchor_points.bind_key("Control-Z", overwrite=True)(self.ctrl_z_callback)
        self.anchor_points.bind_key("Control-+", overwrite=True)(lambda _: extend_mask(self.labels.layer))
        self.anchor_points.bind_key("Control--", overwrite=True)(lambda _: reduce_mask(self.labels.layer))
        self.anchor_points.bind_key("Q", overwrite=True)(lambda _: decrement_selected_label(self.labels.layer))
        self.anchor_points.bind_key("E", overwrite=True)(lambda _: increment_selected_label(self.labels.layer))
        self.anchor_points.bind_key("A", overwrite=True)(scroll_to_prev(self.viewer))
        self.anchor_points.bind_key("D", overwrite=True)(scroll_to_next(self.viewer))
        self.anchor_points.bind_key("W", overwrite=True)(lambda _: increase_brush_size(self.labels.layer))
        self.anchor_points.bind_key("S", overwrite=True)(lambda _: decrease_brush_size(self.labels.layer))
        if self.on_double_click not in self.anchor_points.mouse_double_click_callbacks:
            self.anchor_points.mouse_double_click_callbacks.append(self.on_double_click)
        if self.on_mouse_move not in self.anchor_points.mouse_move_callbacks:
            self.anchor_points.mouse_move_callbacks.append(self.on_mouse_move)
        if self.on_right_click not in self.anchor_points.mouse_drag_callbacks:
            self.anchor_points.mouse_drag_callbacks.insert(0, self.on_right_click)
        if self.on_mouse_wheel not in self.anchor_points.mouse_wheel_callbacks:
            self.anchor_points.mouse_wheel_callbacks.append(self.on_mouse_wheel)

    def add_mode_only(self, event):
        if event.mode not in ["add", "pan_zoom"]:
            warnings.warn("Cannot change mode to %s: only 'add' and 'pan_zoom' mode is allowed" % event.mode)
            event.source.mode = "add"

    def delayed_set_image(self, *args):
        if self.prev_timer_id is not None:
            self.killTimer(self.prev_timer_id)
        self.prev_timer_id = self.startTimer(1000)

    def timerEvent(self, *args, **kwargs):
        self.apply_blurring()
        self.set_image()
        self.killTimer(self.prev_timer_id)
        self.prev_timer_id = None

    def store_orig_image(self):
        image_layer: Image = self.image.layer
        if image_layer != self._prev_img_layer:
            if self._prev_img_layer is not None:
                self._prev_img_layer.data = self._orig_image
            self._orig_image = image_layer.data.copy() if image_layer is not None else None
        self._prev_img_layer = image_layer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qt_viewer = self.viewer.window.qt_viewer
        if image_layer in qt_viewer.controls.widgets:
            controls = qt_viewer.controls.widgets[image_layer]
            for child in controls.children():
                if type(child) == _QDoubleRangeSlider:
                    try:
                        child.sliderReleased.connect(self.on_contrast_slider_released, Qt.UniqueConnection)
                    except TypeError:
                        pass
                    break

    def set_image(self, *args):
        image_layer: Image = self.image.layer
        if image_layer is None:
            self._img = None
            self.feature_editor.image = None
            self.feature_editor.features = None
            return
        if self.viewer.dims.ndisplay == 3:
            return
        if not image_layer.visible:
            image_layer.set_view_slice()
        self.autoincrease_label_id_checkbox.setChecked(image_layer.ndim == 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = image_layer._data_view.astype(float)
        if image.ndim == 2:
            image = np.concatenate([image[..., np.newaxis]] * 3, axis=2)
        min_, max_ = image_layer.contrast_limits if self.apply_contrast_limits else (image.min(), image.max())
        image = np.clip((image - min_) / (max_ - min_), 0, 1)
        if self.feature_inverted:
            image = 1 - image
        self._img = image
        if self.blur_image_checkbox.isChecked():
            image = self.blur_image(image).astype(float)
        if self.feature_dropdown.currentText() == CUSTOM_TEXT:
            self.feature_editor.image = image
            self.feature_editor.execute()
        else:
            grad_x, grad_y = self.feature_manager.get_features(self.image.layer)
            if grad_x is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    grad_x = grad_x.transpose(image_layer._get_order()[:2])
                    grad_y = grad_y.transpose(image_layer._get_order()[:2])
                if self.feature_dropdown.currentText() == SYM_GRADIENT_TEXT:
                    grad_x, grad_y = grad_x / (max_ - min_), grad_y / (max_ - min_)
                    grad_magnitude = np.linalg.norm([grad_x, grad_y], axis=0)
                    min_, max_ = grad_magnitude.min(), grad_magnitude.max()
                    grad_magnitude = (grad_magnitude - min_) / (max_ - min_)
                    self.calculator.set_image(np.concatenate([grad_magnitude[..., None]]*3, axis=-1), np.empty((0,0)), np.empty((0,0)))
                else:
                    self.calculator.set_image(image, grad_x, grad_y)
        self.anchor_points.translate = image_layer.translate
        self.from_e_points_layer.translate = image_layer.translate
        self.to_s_points_layer.translate = image_layer.translate
        self.output.translate = image_layer.translate

    def data_event(self, event):
        if event.source != self.anchor_points:
            return
        if self.remove_last_anchor:
            with self.anchor_points.events.data.blocker():
                self.anchor_points.data = self.anchor_points.data[:-1]
                self.remove_last_anchor = False
                return
        anchor_data = self.anchor_points.data # TODO check
        if len(anchor_data) < self.prev_n_anchor_points:
            with self.anchor_points.events.data.blocker():
                self.clear_all()
            warnings.warn("Cannot delete a single point. Cleared all anchor points")
            return
        self.prev_n_anchor_points = len(anchor_data)
        if len(anchor_data) == 0:
            return
        anchor_data[-1, list(layer_dims_displayed(self.anchor_points))] = np.clip(anchor_data[-1, list(layer_dims_displayed(self.anchor_points))], 0, np.subtract(self.image.layer.data.shape[:self.image.layer.ndim], 1)[list(layer_dims_displayed(self.image.layer))])
        if len(anchor_data) > 1 and np.all(np.round(anchor_data[-1]) == np.round(anchor_data[-2])):
            with self.anchor_points.events.data.blocker():
                self.anchor_points.data = self.anchor_points.data[:-1]
            self.anchor_points.refresh()
            return
        self.anchor_points.refresh()
        if self.shift_down:
            self.last_added_with_shift = True
            with self.anchor_points.events.data.blocker():
                self.anchor_points.data = np.roll(self.anchor_points.data, 1, 0)
            if len(self.to_s_points_layer.data):
                self.output.data = np.concatenate([self.to_s_points_layer.data, self.output.data], 0)
                self.last_segment_length = len(self.to_s_points_layer.data)
                self.to_s_points_layer.data = np.empty((0, 3))
        else:
            if len(self.from_e_points_layer.data):
                self.output.data = np.concatenate([self.output.data, self.from_e_points_layer.data], 0)
                self.last_added_with_shift = False
                self.last_segment_length = len(self.from_e_points_layer.data)
                self.from_e_points_layer.data = np.empty((0, 3))

        self.point_triangle[-1] = self.anchor_points.data[0][list(layer_dims_displayed(self.anchor_points))]
        self.point_triangle[0] = self.anchor_points.data[-1][list(layer_dims_displayed(self.anchor_points))]

    def on_double_click(self, *args):
        if self.shift_down and len(self.from_e_points_layer.data):
            self.output.data = np.concatenate([self.output.data, self.from_e_points_layer.data], 0)
        elif not self.shift_down and len(self.to_s_points_layer.data):
            self.output.data = np.concatenate([self.to_s_points_layer.data, self.output.data], 0)
        if self.smooth_contour_checkbox.isChecked():
            self.output.data = self.smooth_fourier(self.output.data)
        self.points_to_mask()

    def on_right_click(self, layer, event: Event):
        if event.button == 2 and layer.mode == Mode.ADD:
            self.remove_last_anchor = True
            if self.last_segment_length is not None:
                with self.anchor_points.events.data.blocker():
                    if self.last_added_with_shift:
                        self.anchor_points.data = self.anchor_points.data[1:]
                        self.output.data = self.output.data[self.last_segment_length:]
                    else:
                        self.anchor_points.data = self.anchor_points.data[:-1]
                        self.output.data = self.output.data[:-self.last_segment_length]
                self.point_triangle[-1] = self.anchor_points.data[0][list(layer_dims_displayed(self.anchor_points))]
                self.point_triangle[0] = self.anchor_points.data[-1][list(layer_dims_displayed(self.anchor_points))]
                self.last_segment_length = None

    def on_contrast_slider_released(self):
        self.set_image()

    class DrawWorker(QObject):
        done = Signal("PyQt_PyObject")
        contour: np.ndarray
        mask_shape: tuple

        def run(self):
            mask = skimage.draw.polygon2mask(self.mask_shape, self.contour)
            mask = skimage.filters.rank.median(mask.astype(np.uint8), np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])).astype(bool)
            self.done.emit(mask)

    def set_mask(self, mask):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = np.nonzero(mask)
            self.prev_vals[self.labels.layer] = self.labels.layer._slice.image.raw[mask]
            change_idx = list(self.viewer.dims.current_step)
            for i in range(2):
                change_idx[self.viewer.dims.displayed[i]] = idx[i]

            self.change_idx[self.labels.layer] = tuple(change_idx)
            self.labels.layer._slice.image.raw[mask] = self.labels.layer.selected_label
        self.labels.layer.events.data()
        self.labels.layer.refresh()
        if self.labels.layer and self.autoincrease_label_id_checkbox.isChecked():
            self.labels.layer.selected_label += 1

    def points_to_mask(self):
        if self.image_data is None or len(self.output.data) == 0:
            return
        skip_drawing = False
        if self.labels.layer is None:
            answer = QMessageBox.question(
                self,
                "Missing Labels layer",
                "Create Labels layer?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if answer == QMessageBox.StandardButton.Yes:
                new_labels = self.viewer.add_labels(np.zeros(self.image.layer.data.shape[:self.image.layer.ndim], dtype=np.uint16))
                self.labels.layer = new_labels
            else:
                skip_drawing = True
        elif not np.array_equal(self.labels.layer.data.shape[:self.labels.layer.ndim], self.image.layer.data.shape[:self.image.layer.ndim]):
            answer = QMessageBox.question(
                self,
                "Shape mismatch",
                "Current labels and image layers have different shape."
                " Do you want to create a new labels layer with the appropriate shape?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if answer == QMessageBox.StandardButton.Yes:
                new_labels = self.viewer.add_labels(
                    np.zeros(self.image.layer.data.shape[:self.image.layer.ndim], dtype=np.uint16))
                self.labels.layer = new_labels
            else:
                skip_drawing = True
        if not skip_drawing:
            if not self.labels.layer.visible:
                self.labels.layer.set_view_slice()
            self.progress_dialog.setVisible(True)
            self.draw_worker.contour = np.asarray([np.asarray(self.labels.layer.world_to_data(self.output.data_to_world(p)))[list(layer_dims_displayed(self.labels.layer))] for p in self.output.data])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.draw_worker.mask_shape = self.labels.layer._data_view.shape
            self.draw_thread.start()
        self.clear_all()

    def smooth_fourier(self, points):
        coefficients=max(3, round(self.smooth_contour_spinbox.value()*len(points)))
        mask_2d = ~np.all(points == points.min(), axis=0)
        points_2d = points[:, mask_2d]
        center = points_2d.mean(0)
        points_2d = points_2d - center
        tformed = scipy.fft.rfft(points_2d, axis=0)
        tformed[0] = 0
        inv_tformed = scipy.fft.irfft(tformed[:coefficients], len(points_2d), axis=0) + center
        points[:, mask_2d] = inv_tformed
        return points

    def extend_mask(self, _):
        if self.labels.layer is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = self.labels.layer._slice.image.raw
        mask = labels == self.labels.layer.selected_label
        mask = binary_dilation(mask)
        labels[mask] = self.labels.layer.selected_label
        self.labels.layer.events.data()
        self.labels.layer.refresh()

    def reduce_mask(self, _):
        if self.labels.layer is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = self.labels.layer._slice.image.raw
        mask = labels == self.labels.layer.selected_label
        eroded_mask = binary_erosion(mask)
        labels[mask & ~eroded_mask] = 0
        self.labels.layer.events.data()
        self.labels.layer.refresh()

    def eventFilter(self, src: 'QObject', event: 'QEvent') -> bool:
        try:
            self.modifiers = event.modifiers()
        except:
            pass
        if src == self.selected_id_label and event.type() == QEvent.Enter:
            pos = QCursor.pos()
        elif event.type() == QEvent.Type.HoverMove:
            pos = src.mapToGlobal(event.pos())
        else:
            return False
        pos.setX(pos.x()+20)
        pos.setY(pos.y()+20)
        self.selected_id_label.move(pos)
        return False

    def update_label_tooltip(self):
        if self.labels.layer is None:
            return
        self.selected_id_label.setText(str(self.labels.layer.selected_label))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bg_color = self.labels.layer._selected_color
        if bg_color is None:
            bg_color = np.ones(4)
        h, s, v = colorsys.rgb_to_hsv(*bg_color[:-1])
        bg_color = (255*bg_color).astype(int)
        color = (np.zeros(3) if v > 0.7 else np.ones(3)*255).astype(int)
        self.selected_id_label.setStyleSheet("background-color: rgb(%d,%d,%d); color: rgb(%d,%d,%d)" % (tuple(bg_color[:-1]) + tuple(color)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.viewer.window.qt_viewer.canvas.native.setFocus()

    def apply_blurring(self):
        if self.image.layer is None:
            return
        self.image.layer.data = self._orig_image.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.blur_image_checkbox.isChecked() and self.blur_image_slider.value() > 0.:
                if self.image.layer.data.ndim-(1 if self.image.layer.rgb else 0)>=3:
                    slice_ = tuple(
                        slice(None) if i in layer_dims_displayed(self.image.layer) else self.viewer.dims.current_step[i]
                        for i in range(self.viewer.dims.ndim)
                    )
                else:
                    slice_ = (slice(None), slice(None))
                try:
                    orig_image = self._orig_image[slice_]
                except IndexError:
                    warnings.warn("Current slice is out of bounds, skipping blurring")
                    return
                if self.image.layer.rgb:
                    max_vals = orig_image.max((0, 1))
                    for i in range(3):
                        if max_vals[i] == 0:
                            continue
                        self.image.layer.data[slice_ + (i,)] = self.blur_image(orig_image[..., i]).astype(self._orig_image.dtype)
                else:
                    self.image.layer.data[slice_] = self.blur_image(self._orig_image[slice_]).astype(self._orig_image.dtype)
                self.image.layer.events.data()
                self.image.layer.refresh()

    def update_demo_image(self, *args):
        if self.image.layer is None:
            demo_shape = (DEMO_SIZE, DEMO_SIZE)
            img = np.zeros((DEMO_SIZE, DEMO_SIZE, 3), np.uint8)
        else:
            im_layer: Image = self.image.layer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = im_layer._data_view.astype(float)
            if all(s > DEMO_SIZE for s in img.shape if s > 3):
                demo_shape = (DEMO_SIZE, DEMO_SIZE)
                img = img[(img.shape[0]-DEMO_SIZE)//2:(img.shape[0]+DEMO_SIZE)//2, (img.shape[0]-DEMO_SIZE)//2:(img.shape[0]+DEMO_SIZE)//2]
            else:
                demo_shape = tuple(s for s in img.shape if s > 3)
            img = self.blur_image(img)
            if not im_layer.rgb:
                max_ = im_layer.contrast_limits[1]
                min_ = im_layer.contrast_limits[0]
                img = np.clip((img - min_) / (max_ - min_), 0, 1)
                img = (im_layer.colormap.map(img.ravel()).reshape(img.shape + (4,))*255)[..., :3].copy()
            img = img.astype(np.uint8)
        image = QImage(img, demo_shape[1], demo_shape[0], demo_shape[1]*3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image, Qt.ImageConversionFlag.ColorOnly)
        self.demo_image.setPixmap(pixmap)

    def blur_image(self, img):
        return gaussian(
            img,
            sigma=self.blur_image_slider.value(),
            preserve_range=True,
            channel_axis=-1 if img.ndim == 3 else None,
            truncate=2.
        )

    def on_param_change(self, val):
        self.calculator.set_param(val)
        self.set_image()

    def swap_selection(self, viewer: napari.Viewer):
        if self.anchor_points != viewer.layers.selection.active:
            viewer.layers.selection.select_only(self.anchor_points)
        elif self.labels.layer is not None and self.labels.layer != viewer.layers.selection.active:
            viewer.layers.selection.select_only(self.labels.layer)
