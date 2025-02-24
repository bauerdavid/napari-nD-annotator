import colorsys
import time
import warnings
from functools import wraps
import numpy as np

import skimage
from skimage.filters import gaussian

import napari
from napari.layers.labels._labels_utils import get_dtype
from napari.layers.labels.labels import _coerce_indices_for_vectorization
from napari.layers.labels._labels_constants import Mode
from napari.layers import Image, Labels
from napari.utils.action_manager import action_manager
from napari.utils._dtype import get_dtype_limits
from napari.utils.events import Event
from napari._qt.layer_controls.qt_image_controls_base import _QDoubleRangeSlider
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.widgets.qt_mode_buttons import QtModeRadioButton, QtModePushButton
from napari._qt.qt_resources import get_current_stylesheet

from qtpy.QtCore import Signal, QObject, QEvent, QThread, Qt
from qtpy.QtWidgets import QLabel, QSizePolicy
from magicclass.serialize import serialize
from magicclass import MagicTemplate, field, abstractapi, magicclass, vfield, set_design
from magicclass.widgets import FreeWidget
from magicgui.widgets import ComboBox, Image as MagicImage

from .minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
from .minimal_contour_overlay.minimal_contour_overlay import MinimalContourOverlay
from ._utils.changeable_color_box import QtChangeableColorBox
from ._utils import ScriptExecuteWidget, ProgressWidget
from ._utils.collapsible_widget import CollapsibleContainerGroup, correct_container_size
from ._utils.callbacks import reduce_mask, extend_mask
from .._helper_functions import layer_dims_order, layer_dims_displayed, layer_slice_indices, \
    layer_get_order
from ..minimal_contour import FeatureManager
from .._napari_version import NAPARI_VERSION
from napari_nd_annotator._widgets.resources import mc_contour_style_path, interpolate_style_path


def delay_function(function=None, delay=0.2):

    def decorator(fn):
        from threading import Timer

        _store: dict = {"timer": None, "last_call": 0.0, "args": (), "kwargs": {}}

        @wraps(fn)
        def delayed(*args, **kwargs):
            _store["args"] = args
            _store["kwargs"] = kwargs
            def call_it():
                _store["timer"] = None
                _store["last_call"] = time.time()
                return fn(*_store["args"], **_store["kwargs"])

            now = time.time()
            if not _store["last_call"] or (now - _store["last_call"]) > delay:
                ret = call_it()
                return ret
            else:
                if _store["timer"] is not None:
                    _store["timer"].cancel()
                _store["timer"] = Timer(delay, call_it)
                _store["timer"].start()
            _store["last_call"] = time.time()
            return None

        return delayed

    return decorator if function is None else decorator(function)  # type: ignore


GRADIENT_BASED = 0
INTENSITY_BASED = 2

ASSYM_GRADIENT = "Gradient [assymetric]"
SYM_GRADIENT = "Gradient [symmetric]"
HIGH_INTENSITY = "High intensity"
LOW_INTENSITY = "Low intensity"
CUSTOM = "Custom"
FEATURE_KEYS = [ASSYM_GRADIENT, SYM_GRADIENT, HIGH_INTENSITY, LOW_INTENSITY, CUSTOM]
FEATURES = {
    ASSYM_GRADIENT: GRADIENT_BASED,
    SYM_GRADIENT: INTENSITY_BASED,
    HIGH_INTENSITY: INTENSITY_BASED,
    LOW_INTENSITY: INTENSITY_BASED,
    CUSTOM: INTENSITY_BASED
}

DEMO_SIZE = 200


class ColorBox(FreeWidget):
    def __init__(self, layer: Labels = None):
        super().__init__()
        self.wdt = QtChangeableColorBox(layer)
        self.set_widget(self.wdt)

    @property
    def layer(self):
        return self.wdt.layer

    @layer.setter
    def layer(self, layer):
        self.wdt.layer = layer


class EventFilter(QObject):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def eventFilter(self, src: 'QObject', event: 'QEvent') -> bool:
        if hasattr(event, "modifiers"):
            self.widget.modifiers = event.modifiers()

        # if src == self.widget.selected_id_label and event.type() == QEvent.Enter:
        #     pos = QCursor.pos()
        # elif event.type() == QEvent.Type.HoverMove:
        #     pos = src.mapToGlobal(event.pos())
        # else:
        #     return False
        # pos.setX(pos.x()+20)
        # pos.setY(pos.y()+20)
        # self.widget.selected_id_label.move(pos)
        return False


@magicclass(name="Minimal Contour", widget_type="scrollable", properties={"x_enabled": False})
class MinimalContourWidget(MagicTemplate):
    image_layer_combobox = field(Image, label="Image")

    @magicclass(widget_type="collapsible", name="Image Features")
    class ImageFeaturesWidget(MagicTemplate):
        ...

    @magicclass(widget_type="collapsible", name="Blurring")
    class BlurWidget(MagicTemplate):
        blur_image_checkbox = abstractapi()
        blur_sigma_slider = abstractapi()

    @magicclass(widget_type="collapsible", name="Contour Options")
    class ContourWidget(MagicTemplate):
        ...

    @magicclass(widget_type="collapsible", name="Label Options")
    class LabelOptionsWidget(MagicTemplate):
        @magicclass(layout="horizontal", name="selected_label_widget", labels=False,)
        class SelectedLabelWidget(MagicTemplate):
            ...

        autoincrease_label_id = abstractapi()

        @magicclass(layout="horizontal", name="label_manipulation_widget")
        class LabelManipulationWidget(MagicTemplate):
            def reduce_mask(self): ...

            def extend_mask(self): ...


    color_box = field(ColorBox, location=LabelOptionsWidget.SelectedLabelWidget)
    selected_label_spinbox = field(int, location=LabelOptionsWidget.SelectedLabelWidget).with_options(tooltip="Selected label for the current 'labels' layer.")
    autoincrease_label_id = vfield(bool, label="Auto-increment Label", location=LabelOptionsWidget).with_options(value=True,
                                                      tooltip="When checked, selected label will be incremented\n"
                                                              "after completing the contour")

    used_feature_combobox = field(widget_type=ComboBox, label="Used Feature", options={"choices": FEATURE_KEYS}, location=ImageFeaturesWidget)
    feature_editor = field(ScriptExecuteWidget, location=ImageFeaturesWidget).with_options(editor_key="minimal_contour_features")
    param_spinbox = field(int, label="Parameter", location=ImageFeaturesWidget).with_options(min=1, max=50, value=5,
                                            tooltip="Increasing this parameter will cause contours to be more aligned\n"
                                                    "with the selected image feature (with a lower number the contour will be\n"
                                                    "closer to the Euclidean shortest path")
    local_intensity_correction_checkbox = field(bool, location=ImageFeaturesWidget).with_options(visible=False)


    blur_image_checkbox = field(bool, label="Blur image", location=BlurWidget).with_options(tooltip="When checked, the image will be blurred\n"
                                        "before calculating minimal contour")
    blur_sigma_slider = field(float, label="Sigma", widget_type="FloatSlider", location=BlurWidget).with_options(max=20, value=1)

    smooth_contour_checkbox = field(bool, label="Use contour smoothing", location=ContourWidget).with_options(
        tooltip="When checked, the finished contour will be smoothed\n"
                "to remove minor noise using Fourier transformation.")
    contour_smoothness_slider = field(float, label="Contour smoothness", widget_type="FloatSlider", location=ContourWidget).with_options(min=0., max=1.,
                                                          tooltip="Number of Fourier coefficients to approximate the contour.\n"
                                                                  "Lower number -> smoother contour\n"
                                                                  "Higher number -> more faithful to the original")
    contour_width_spinbox = field(int, label="Contour width", location=ContourWidget).with_options(min=1, max=50, value=2, tooltip="Point size for contour display.")

    def __init__(self, viewer: napari.Viewer = None):
        self.progress_dialog = ProgressWidget(self.native, message="Drawing mask...")
        self.draw_worker = self.DrawWorker()
        self.draw_thread = QThread()
        self.draw_worker.moveToThread(self.draw_thread)
        self.draw_worker.done.connect(self.draw_thread.quit)
        self.draw_worker.done.connect(self._set_mask)
        self.draw_worker.done.connect(lambda: self.progress_dialog.setVisible(False))
        self.draw_thread.started.connect(self.draw_worker.run)
        self._img = None
        self._features = None
        self._viewer: napari.Viewer = viewer
        self._labels_layer: Labels = None
        self.apply_contrast_limits = True
        self._prev_img_layer = self.image_layer
        self._orig_image = self.image_layer.data if self.image_layer else None
        self.point_triangle = np.zeros((3, 2), dtype=np.float64) - 1  # start point, current position, end point
        self.remove_last_anchor = False
        self.last_added_with_shift = None
        self.last_segment_length = None
        self.prev_n_anchor_points = 0
        self.prev_labels_layer = self.labels_layer
        self.feature_inverted = False
        self.prev_timer_id = None
        self.feature_manager = None
        self.event_filter = EventFilter(self)

        # Ctrl+Z handling

        self.selected_id_label = QLabel(parent=self.native)
        self.selected_id_label.setFixedSize(20, 20)
        self.selected_id_label.setAlignment(Qt.AlignCenter)
        self.selected_id_label.setSizePolicy(QSizePolicy.Fixed, self.selected_id_label.sizePolicy().verticalPolicy())

        self.selected_id_label.setWindowFlags(Qt.Window
                                              | Qt.FramelessWindowHint
                                              | Qt.WindowStaysOnTopHint)
        self.selected_id_label.setAttribute(Qt.WA_ShowWithoutActivating)
        self.selected_id_label.installEventFilter(self.event_filter)
        self._collapsible_group = CollapsibleContainerGroup()

        self.demo_image_widget = MagicImage()
        self.demo_image_widget.native.setFixedSize(DEMO_SIZE, DEMO_SIZE)
        self.demo_image_widget.native.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.demo_image_widget.native.setWindowFlags(Qt.Window
                                              | Qt.FramelessWindowHint
                                              | Qt.WindowStaysOnTopHint)
        self.demo_image_widget.native.setAttribute(Qt.WA_ShowWithoutActivating)

        self.modifiers = None

    def __post_init__(self):
        VispyMinimalContourOverlay.mc_calculator.set_param(self.param)
        self.local_intensity_correction_checkbox.changed.connect(VispyMinimalContourOverlay.mc_calculator.set_use_local_maximum)

        if self.labels_layer is not None:
            dtype_lims = get_dtype_limits(get_dtype(self.labels_layer))
        else:
            dtype_lims = 0, np.iinfo(int).max
        selected_label_spinbox = self.selected_label_spinbox
        selected_label_spinbox.min, selected_label_spinbox.max = dtype_lims
        self._set_use_smoothing(self.is_blurring_enabled, update_image=False)
        self._on_feature_change(self.used_feature, update_image=False)
        self._on_param_change(self.param, update_image=False)
        self.feature_editor.script_worker.done.connect(self._set_features)

        self.native.setMinimumWidth(self.native.children()[1].widget().sizeHint().width()
                                    + self.native.children()[1].verticalScrollBar().sizeHint().width()
                                    + 2 * self.native.children()[1].widget().layout().getContentsMargins()[2])
        for container in [self.ImageFeaturesWidget, self.BlurWidget, self.ContourWidget, self.LabelOptionsWidget]:
            container._widget._expand_btn.setChecked(False)
            # container._widget._inner_qwidget.setStyleSheet("padding: 0 10 0 10")
            # container._widget._inner_qwidget.setSizePolicy(QSizePolicy.Minimum, container._widget._inner_qwidget.sizePolicy().verticalPolicy())
            container.collapsed = True
            self._collapsible_group.addItem(container)
        self.feature_editor.visible = self.used_feature == CUSTOM
        self.feature_editor.run_button.setEnabled(self.image_layer is not None)
        self.feature_editor.run_button.setToolTip("Open an image first" if self.image_layer is None else "")
        self.param_spinbox.visible = self.used_feature == ASSYM_GRADIENT
        self.demo_image_widget.native.setFixedSize(DEMO_SIZE, DEMO_SIZE)
        self.demo_image_widget.native.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.contour_smoothness_slider.visible = self.is_contour_smoothing_enabled
        self.selected_label_spinbox.native.setKeyboardTracking(False)
        self.selected_label_spinbox.native.setAlignment(Qt.AlignCenter)
        self.blur_sigma_slider.native.children()[0].sliderPressed.connect(self._show_demo_image)
        self.blur_sigma_slider.native.children()[0].sliderReleased.connect(self._hide_demo_image)
        self.blur_sigma_slider.native.children()[0].sliderReleased.connect(self._apply_blurring)
        self.blur_sigma_slider.native.children()[0].sliderReleased.connect(self._set_image)
        self._on_label_change()
        self._on_selected_label_change()
        if self._viewer is not None:
            self._initialize(self._viewer)

    def __magicclass_serialize__(self):
        d = serialize(self)
        if "image_layer_combobox" in d:
            del d["image_layer_combobox"]
        if "labels_layer_combobox" in d:
            del d["labels_layer_combobox"]
        return d

    @property
    def viewer(self):
        return self._viewer

    @property
    def image_layer(self):
        return self.image_layer_combobox.value

    @image_layer.setter
    def image_layer(self, new_layer):
        self.image_layer_combobox.value = new_layer

    @property
    def image_data(self):
        return self._img

    @property
    def labels_layer(self) -> Labels:
        return self._labels_layer

    @property
    def blur_sigma(self):
        return self.blur_sigma_slider.value

    @property
    def used_feature(self):
        return self.used_feature_combobox.value

    @property
    def param(self):
        return self.param_spinbox.value

    @property
    def is_blurring_enabled(self):
        return self.blur_image_checkbox.value

    @property
    def is_contour_smoothing_enabled(self):
        return self.smooth_contour_checkbox.value

    @property
    def contour_smoothness(self):
        return self.contour_smoothness_slider.value

    @property
    def contour_width(self):
        return self.contour_width_spinbox.value

    @property
    def shift_down(self):
        return self.modifiers & Qt.ShiftModifier if self.modifiers is not None else False

    @property
    def ctrl_down(self):
        return self.modifiers & Qt.ControlModifier if self.modifiers is not None else False

    @property
    def alt_down(self):
        return self.modifiers & Qt.AltModifier if self.modifiers is not None else False

    def _initialize_helper_layers(self):
        self._change_contour_width(self.contour_width)

    class DrawWorker(QObject):
        done = Signal("PyQt_PyObject")
        contour: np.ndarray
        mask_shape: tuple

        def run(self):
            mask = skimage.draw.polygon2mask(self.mask_shape, self.contour)
            mask = skimage.filters.rank.median(mask.astype(np.uint8), np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])).astype(bool)
            self.done.emit(mask)

    def _set_mask(self, mask):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = np.nonzero(mask)
            labels_layer = self.labels_layer
            ndim = labels_layer.data.ndim
            change_idx = np.zeros((ndim, len(idx[0])), dtype=int)
            order = layer_dims_order(labels_layer)
            dims_displayed = layer_dims_displayed(labels_layer)
            slice_indices = layer_slice_indices(labels_layer)
            for i, d in enumerate(order):
                change_idx[d] = idx[i-ndim+2] if d in dims_displayed else slice_indices[d]
            # change_idx = [idx[i] if i in dims_displayed else slice_indices[i] for i in range(ndim)]
            change_idx = _coerce_indices_for_vectorization(labels_layer.data, change_idx)
            if hasattr(labels_layer, "data_setitem"):
                labels_layer.data_setitem(change_idx, labels_layer.selected_label)
            else:
                labels_layer._save_history(
                    (change_idx, labels_layer._slice.image.raw[mask], labels_layer.selected_label))
                labels_layer._slice.image.raw[mask] = labels_layer.selected_label
                labels_layer.events.data()
                labels_layer.refresh()
        if labels_layer and self.autoincrease_label_id:
            labels_layer.selected_label += 1

    def _apply_blurring(self):
        image_layer = self.image_layer
        if image_layer is None:
            return
        image_layer.data = self._orig_image.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.is_blurring_enabled and self.blur_sigma > 0.:
                if image_layer.data.ndim-(1 if image_layer.rgb else 0)>=3:
                    slice_ = tuple(
                        slice(None) if i in layer_dims_displayed(self.image_layer) else self.viewer.dims.current_step[i]
                        for i in range(self.viewer.dims.ndim)
                    )
                else:
                    slice_ = (slice(None), slice(None))
                try:
                    orig_image = self._orig_image[slice_]
                except IndexError:
                    warnings.warn("Current slice is out of bounds, skipping blurring")
                    return
                if image_layer.rgb:
                    max_vals = orig_image.max((0, 1))
                    for i in range(3):
                        if max_vals[i] == 0:
                            continue
                        image_layer.data[slice_ + (i,)] = self._blur_image(orig_image[..., i]).astype(self._orig_image.dtype)
                else:
                    image_layer.data[slice_] = self._blur_image(self._orig_image[slice_]).astype(self._orig_image.dtype)
                image_layer.events.data()
                image_layer.refresh()

    # tooltip = "Reduce the selected label by one pixel around the borders\n"
    # "using erosion"
    @set_design(location=LabelOptionsWidget.LabelManipulationWidget)
    def reduce_mask(self):
        if self.labels_layer is not None:
            reduce_mask(self.labels_layer)

    # tooltip = "Extend the selected label by one pixel around the borders\n"
    # "using dilation"
    @set_design(location=LabelOptionsWidget.LabelManipulationWidget)
    def extend_mask(self):
        if self.labels_layer is not None:
            extend_mask(self.labels_layer)

    @image_layer_combobox.connect
    def _on_image_changed(self, *args):
        self._store_orig_image(*args)
        self.feature_editor.run_button.setEnabled(self.image_layer is not None)
        self.feature_editor.run_button.setToolTip("Open an image first" if self.image_layer is None else "")
        self._set_image("_on_image_changed")

    # @labels_layer_combobox.connect
    def _on_label_change(self, _=None):
        labels_layer = self.labels_layer
        selected_label_spinbox = self.selected_label_spinbox
        colorbox = self.color_box
        if labels_layer is not None:
            dtype_lims = get_dtype_limits(get_dtype(labels_layer))
            selected_label_spinbox.value = labels_layer.selected_label
        else:
            dtype_lims = 0, np.iinfo(int).max
        selected_label_spinbox.min, selected_label_spinbox.max = dtype_lims
        colorbox.layer = labels_layer
        if self.prev_labels_layer is not None:
            self.prev_labels_layer.events.selected_label.disconnect(self._on_selected_label_change)
            if self._on_mouse_wheel in self.prev_labels_layer.mouse_wheel_callbacks:
                self.prev_labels_layer.mouse_wheel_callbacks.remove(self._on_mouse_wheel)
            # self.prev_labels_layer.bind_key("Shift", overwrite=True)(None)
            self.prev_labels_layer.bind_key("Alt", overwrite=True)(None)
            self.prev_labels_layer.bind_key("Control-+", overwrite=True)(None)
            self.prev_labels_layer.bind_key("Control--", overwrite=True)(None)
        if labels_layer is not None:
            labels_layer.events.selected_label.connect(
                self._on_selected_label_change
            )
            if self._on_mouse_wheel not in labels_layer.mouse_wheel_callbacks:
                labels_layer.mouse_wheel_callbacks.append(self._on_mouse_wheel)
            labels_layer.bind_key("Shift", overwrite=True)(self._shift_pressed)
            self._on_selected_label_change()
        self.prev_labels_layer = labels_layer

    @used_feature_combobox.connect
    def _on_feature_change(self, used_feature, update_image=True):
        self.feature_editor.visible = (used_feature == CUSTOM)
        self.feature_inverted = (used_feature == LOW_INTENSITY)
        VispyMinimalContourOverlay.mc_calculator.set_method(FEATURES[used_feature])
        self.param_spinbox.visible = used_feature == ASSYM_GRADIENT
        if update_image:
            self._set_image("_on_feature_change")

    @param_spinbox.connect
    def _on_param_change(self, val, update_image=True):
        VispyMinimalContourOverlay.mc_calculator.set_param(val)
        if update_image:
            self._set_image("_on_param_change")

    @blur_image_checkbox.connect
    def _set_use_smoothing(self, _, update_image=True):
        self._apply_blurring()
        if update_image:
            self._set_image("_set_use_smoothing")

    def _show_demo_image(self):
        pos = self.blur_sigma_slider.native.pos()
        pos.setX(min(pos.x(), self.native.size().width()-(DEMO_SIZE+10)))
        pos.setY(pos.y() + DEMO_SIZE)
        pos = self.native.mapToGlobal(pos)
        self.demo_image_widget.native.move(pos)
        self.demo_image_widget.visible = True

    def _hide_demo_image(self):
        self.demo_image_widget.visible = False

    @blur_sigma_slider.connect
    def _update_demo_image(self, event=None):
        im_layer: Image = self.image_layer
        if im_layer is None:
            img = np.ones((DEMO_SIZE, DEMO_SIZE, 3), np.uint8)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # img = im_layer._data_view.astype(float)
                im_layer._set_view_slice()
                slice_indices = layer_slice_indices(im_layer)
                img = self._orig_image[slice_indices].astype(float)
            if all(s > DEMO_SIZE for s in (img.shape[:2] if im_layer.rgb else img.shape)):
                img = img[(img.shape[0]-DEMO_SIZE)//2:(img.shape[0]+DEMO_SIZE)//2, (img.shape[0]-DEMO_SIZE)//2:(img.shape[0]+DEMO_SIZE)//2]
            if self.is_blurring_enabled:
                img = self._blur_image(img)
            if not im_layer.rgb:
                max_ = im_layer.contrast_limits[1]
                min_ = im_layer.contrast_limits[0]
                img = np.clip((img - min_) / (max_ - min_), 0, 1)
                img = (im_layer.colormap.map(img.ravel()).reshape(img.shape + (4,))*255)[..., :3].copy()
            img = img.astype(np.uint8)
        self.demo_image_widget.value = img

    @smooth_contour_checkbox.connect
    def _on_smooth_checked(self, is_checked):
        self.contour_smoothness_slider.visible = is_checked
        correct_container_size(self.ContourWidget)
        if self.labels_layer is not None:
            self.labels_layer._overlays["minimal_contour"].contour_smoothness = self.contour_smoothness if is_checked else 1.

    @contour_smoothness_slider.connect
    def _on_smoothness_changed(self, value):
        if self.labels_layer is not None:
            self.labels_layer._overlays["minimal_contour"].contour_smoothness = value

    @contour_width_spinbox.connect
    def _change_contour_width(self, value):
        if self.labels_layer is not None:
            self.labels_layer._overlays["minimal_contour"].contour_width = value

    @selected_label_spinbox.connect
    def _change_selected_label(self, value):
        if self.labels_layer is None:
            if self.selected_label_spinbox.value != 0:
                self.selected_label_spinbox.value = 0
            return
        self.labels_layer.selected_label = value
        self.LabelOptionsWidget.SelectedLabelWidget.native.clearFocus()
        self.native.setFocus()

    def _on_selected_label_change(self):
        """Receive layer model label selection change event and update spinbox."""
        labels_layer = self.labels_layer
        if labels_layer is None:
            return
        with labels_layer.events.selected_label.blocker():
            value = labels_layer.selected_label
            self.selected_label_spinbox.value = value
            self._update_label_tooltip()

    def _shift_pressed(self, *_):
        # TODO handle correctly
        return
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     if self.labels_layer is not None and len(self.anchor_points.data) == 0 and self.viewer.window.qt_viewer.canvas.native.hasFocus():
        #         self.selected_id_label.show()
        # self.from_e_points_layer.face_color = "gray"
        # self.from_e_points_layer.current_face_color = "gray"
        # self.to_s_points_layer.face_color = "red"
        # self.to_s_points_layer.current_face_color = "red"
        # self.from_e_points_layer.edge_color = "gray"
        # self.from_e_points_layer.current_edge_color = "gray"
        # self.to_s_points_layer.edge_color = "red"
        # self.to_s_points_layer.current_edge_color = "red"
        # yield
        # self.selected_id_label.hide()
        # self.from_e_points_layer.face_color = "red"
        # self.from_e_points_layer.current_face_color = "red"
        # self.to_s_points_layer.face_color = "gray"
        # self.to_s_points_layer.current_face_color = "gray"
        # self.from_e_points_layer.edge_color = "red"
        # self.from_e_points_layer.current_edge_color = "red"
        # self.to_s_points_layer.edge_color = "gray"
        # self.to_s_points_layer.current_edge_color = "gray"

    def _ctrl_shift_pressed(self, _):
        shift_callback = self._shift_pressed(_)
        yield from shift_callback
        yield from shift_callback

    def _ctrl_z_callback(self, _):
        if self.labels_layer is not None and hasattr(self.labels_layer, "data_setitem"):
            self.labels_layer.undo()

    def _on_mouse_wheel(self, _, event):
        labels_layer = self.labels_layer
        if labels_layer is None:
            return
        delta = (np.sign(event.delta)).astype(int)
        if self.shift_down:
            labels_layer.selected_label = max(0, labels_layer.selected_label + delta[1])
        elif self.alt_down:
            diff = delta[0]
            diff *= min(max(1, labels_layer.brush_size//10), 5)
            labels_layer.brush_size = max(1, labels_layer.brush_size + diff)

    def _on_right_click(self, layer, event: Event):
        # TODO remove
        return
        if event.button != 2 or layer.mode != Mode.ADD:
            return
        self.remove_last_anchor = True
        if self.last_segment_length is None:
            return
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

    def _blur_image(self, img):
        return gaussian(
            img,
            sigma=self.blur_sigma,
            preserve_range=True,
            channel_axis=-1 if img.ndim == 3 else None,
            truncate=2.
        )

    def _store_orig_image(self, *_):
        image_layer: Image = self.image_layer
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
                        child.sliderReleased.connect(self._set_image, Qt.UniqueConnection)
                    except TypeError:
                        pass
                    break

    def _set_image(self, event=None):
        image_layer: Image = self.image_layer
        if image_layer is None:
            self._img = None
            self.feature_editor.variables["image"] = None
            VispyMinimalContourOverlay.set_calculator_feature(None)
            # self.feature_editor.features = None #TODO check
            return
        if self.viewer is None:
            return
        if self.viewer.dims.ndisplay == 3:
            # self.anchor_points.editable = False
            return
        self._apply_blurring()
        if not image_layer.visible:
            image_layer.set_view_slice()
        if NAPARI_VERSION < "0.5.0":
            image_layer._slice_dims(self.viewer.dims.point, self.viewer.dims.ndisplay, self.viewer.dims.order)
        else:
            image_layer._slice_dims(self.viewer.dims)
        self.autoincrease_label_id = (image_layer.ndim == 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = image_layer._data_view.astype(float)
        if image.ndim == 2:
            image = np.concatenate([image[..., np.newaxis]] * 3, axis=2)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        min_, max_ = image_layer.contrast_limits if self.apply_contrast_limits else (image.min(), image.max())
        image = np.clip((image - min_) / (max_ - min_), 0, 1)
        if self.feature_inverted:
            image = 1 - image
        self._img = image
        if self.is_blurring_enabled:
            image = self._blur_image(image).astype(float)
        if self.used_feature == CUSTOM:
            self.feature_editor.variables["image"] = image
            self.feature_editor.Run()
        else:
            grad_x, grad_y = self.feature_manager.get_features(image_layer)
            if grad_x is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    order = layer_get_order(image_layer)
                    grad_x = grad_x.transpose(order[:2])
                    grad_y = grad_y.transpose(order[:2])
                if self.is_blurring_enabled:
                    grad_x = self._blur_image(grad_x)
                    grad_y = self._blur_image(grad_y)
                if self.used_feature == SYM_GRADIENT:
                    grad_x, grad_y = grad_x / (max_ - min_), grad_y / (max_ - min_)
                    grad_magnitude = np.linalg.norm([grad_x, grad_y], axis=0)
                    min_, max_ = grad_magnitude.min(), grad_magnitude.max()
                    grad_magnitude = (grad_magnitude - min_) / (max_ - min_)
                    VispyMinimalContourOverlay.set_calculator_feature(grad_magnitude)
                else:
                    VispyMinimalContourOverlay.set_calculator_feature(image, grad_x, grad_y)
        # self.anchor_points.translate = image_layer.translate
        # self.from_e_points_layer.translate = image_layer.translate
        # self.to_s_points_layer.translate = image_layer.translate
        # self.output.translate = image_layer.translate

    @delay_function(delay=.5)
    def _delayed_set_image(self, *args):
        self._set_image(args)

    def _set_features(self, var_dict: dict):
        if "exception" in var_dict:
            raise var_dict["exception"]
        features = var_dict.get("features", None)
        if features is None:
            warnings.warn("The 'features' variable was not set in the script.")
            return
        if features.ndim == 2:
            features = np.concatenate([features[..., np.newaxis]] * 3, -1)
        self._features = features
        VispyMinimalContourOverlay.set_calculator_feature(features.astype(float))

    def _update_label_tooltip(self):
        labels_layer = self.labels_layer
        if labels_layer is None:
            return
        self.selected_id_label.setText(str(labels_layer.selected_label))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bg_color = labels_layer._selected_color
        if bg_color is None:
            bg_color = np.ones(4)
        h, s, v = colorsys.rgb_to_hsv(*bg_color[:-1])
        bg_color = (255*bg_color).astype(int)
        color = (np.zeros(3) if v > 0.7 else np.ones(3)*255).astype(int)
        self.selected_id_label.setStyleSheet("background-color: rgb(%d,%d,%d); color: rgb(%d,%d,%d)" % (tuple(bg_color[:-1]) + tuple(color)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.viewer.window.qt_viewer.canvas.native.setFocus()

    def _on_active_layer_changed(self, *_):
        active_layer = self.viewer.layers.selection.active
        self._labels_layer = active_layer if isinstance(active_layer, Labels) else None
        if self._labels_layer:
            if "minimal_contour" not in self._labels_layer._overlays:
                self._labels_layer.bind_key("Control", self._on_ctrl_pressed)
                self._labels_layer._overlays.update({"minimal_contour": MinimalContourOverlay()})
                labels_control: QtLabelsControls = self.viewer.window.qt_viewer.controls.widgets[self._labels_layer]
                mc_btn = QtModeRadioButton(self._labels_layer, "minimal contour", Mode.PAN_ZOOM)
                action_manager.bind_button(
                    'napari-nD-annotator:activate_labels_mc_mode',
                    mc_btn,
                    # extra_tooltip_text=extra_tooltip_text,
                )
                for button in labels_control.button_group.buttons():
                    button.toggled.connect(self._disable_mc_mode)
                mc_btn.setStyleSheet(get_current_stylesheet([mc_contour_style_path]))
                labels_control.button_group.addButton(mc_btn)
                labels_control._EDIT_BUTTONS += (mc_btn,)
                labels_control.mc_button = mc_btn
                labels_control.button_grid.addWidget(labels_control.mc_button, 1, 0)

                def switch_to_mc_mode(*_, **__):
                    mc_btn.blockSignals(True)
                    mc_btn.setChecked(True)
                    mc_btn.blockSignals(False)
                    self._enable_mc_mode()
                    # btn.setChecked(True)
                self._labels_layer.bind_key("0", switch_to_mc_mode)
            self.labels_layer._overlays["minimal_contour"].contour_smoothness = self.contour_smoothness if self.is_contour_smoothing_enabled else 1.

    def _on_ctrl_pressed(self, _):
        if self.labels_layer is None:
            return
        self.labels_layer._overlays["minimal_contour"].use_straight_lines = True
        yield
        self.labels_layer._overlays["minimal_contour"].use_straight_lines = False


    def _set_viewer_callbacks(self):
        self.viewer.layers.events.connect(self.image_layer_combobox.reset_choices)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

        def change_layer_callback(num):
            def change_layer(_):
                if num - 1 < len(self.viewer.layers):
                    self.viewer.layers.selection.select_only(self.viewer.layers[num - 1])
                else:
                    warnings.warn("%d is out of range (number of layers: %d)" % (num, len(self.viewer.layers)))

            return change_layer

        for i in range(1, 10):
            self.viewer.bind_key("Control-%d" % i, overwrite=True)(change_layer_callback(i))
        self.viewer.dims.events.current_step.connect(self._delayed_set_image, position="last")

        self.viewer.dims.events.ndisplay.connect(self._set_image, position="last")
        self.viewer.dims.events.order.connect(self._set_image, position="last")
        self.viewer.layers.events.removed.connect(lambda e: self.feature_manager.remove_features(e.value))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.viewer.window.qt_viewer.window().installEventFilter(self.event_filter)
            self.viewer.window.qt_viewer.window().setMouseTracking(True)

    def _initialize(self, viewer: napari.Viewer):
        self._viewer = viewer
        self._initialize_helper_layers()
        self.feature_manager = FeatureManager(self.viewer)
        self._set_viewer_callbacks()
        self._store_orig_image()
        self._apply_blurring()
        self._update_label_tooltip()
        self._set_image()
        action_manager.register_action("napari-nD-annotator:activate_labels_mc_mode", self._enable_mc_mode,
                                       "We're switching to MC mode", None)

    def _enable_mc_mode(self, *args, **kwargs):
        if self.labels_layer is not None:
            self.labels_layer._overlays["minimal_contour"].enabled = True

    def _disable_mc_mode(self, state):
        if state and self.labels_layer is not None:
            self.labels_layer._overlays["minimal_contour"].enabled = False
