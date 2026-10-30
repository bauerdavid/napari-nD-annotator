import colorsys
import os
import time
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import numpy as np

import skimage
from skimage.filters import gaussian

import napari
from napari.layers.labels._labels_utils import get_dtype
from napari.layers.labels._labels_constants import Mode
from napari.layers import Image, Labels
from napari.utils.action_manager import action_manager
from napari.utils._dtype import get_dtype_limits
from superqt.sliders import QDoubleRangeSlider
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.widgets.qt_mode_buttons import QtModeRadioButton, QtModePushButton
from napari._qt.qt_resources import get_current_stylesheet

from qtpy.QtCore import Signal, QObject, QEvent, QThread, Qt, QTimer
from qtpy.QtWidgets import QLabel, QSizePolicy, QMessageBox, QComboBox, QDialog, QDialogButtonBox, QVBoxLayout
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
from ._utils.shortcuts import LABELS, VIEWER, bind_all
from .._helper_functions import layer_dims_order, layer_dims_displayed, layer_dims_not_displayed, layer_ndisplay, \
    layer_slice_indices, _coerce_indices_for_vectorization, viewer_qt_viewer, unwrap_viewer, \
    viewer_main_window, viewer_camera
from ..minimal_contour import roi as mc_roi
from ..minimal_contour._legacy_tmp import clean_legacy_tmp
from ..minimal_contour.intensity_prior import MAX_SLOWDOWN as MAX_PRIOR_SLOWDOWN, SIMILAR, SIMILAR_OR_BRIGHTER, \
    SIMILAR_OR_DARKER
from ..minimal_contour.plane_features import PlaneFeatureCache, to_float64_pair
from .object_list import selected_object
from napari_nd_annotator._widgets.resources import mc_contour_style_path, interpolate_style_path


class _FeatureBridge(QObject):
    """Delivers a worker thread's result to the GUI thread (queued connection)."""
    ready = Signal(int, object)


def delay_function(function=None, delay=0.2, leading=True):
    """Rate-limit a function; with ``leading=False`` it only runs ``delay`` seconds after the last call."""

    def decorator(fn):
        _store: dict = {"timer": None, "last_call": 0.0, "args": (), "kwargs": {}}

        def call_it():
            try:
                return fn(*_store["args"], **_store["kwargs"])
            finally:
                # stamped when fn is done: if fn takes longer than the delay, stamping at its start would make the
                # next event (e.g. the next step of a dragged slider) look idle and run fn again right away
                _store["last_call"] = time.time()

        @wraps(fn)
        def delayed(*args, **kwargs):
            _store["args"] = args
            _store["kwargs"] = kwargs

            now = time.time()
            if leading and (not _store["last_call"] or (now - _store["last_call"]) > delay):
                return call_it()
            if _store["timer"] is None:
                # QTimer fires on the GUI thread. A threading.Timer would run fn
                # on a worker thread, where refreshing a layer leaves the canvas
                # unrepainted until something else triggers a redraw.
                timer = QTimer()
                timer.setSingleShot(True)
                timer.timeout.connect(call_it)
                _store["timer"] = timer
            _store["timer"].start(int(delay * 1000))
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
# set this environment variable (to anything but "" or "0") to show the experimental intensity prior controls
INTENSITY_PRIOR_ENV = "ND_ANNOTATOR_INTENSITY_PRIOR"
# how many times slower the pixels least like the anchors' intensity get (see minimal_contour/intensity_prior.py)
PRIOR_SLOWDOWN_CHOICES = [("×%g" % s, s) for s in (3., 10., 30., 100., 300., MAX_PRIOR_SLOWDOWN)]
DEFAULT_PRIOR_SLOWDOWN = 100.
DEFAULT_PRIOR_FEATURE_WEIGHT = 0.25
PRIOR_MATCH_CHOICES = [("Similar", SIMILAR), ("Similar or brighter", SIMILAR_OR_BRIGHTER),
                       ("Similar or darker", SIMILAR_OR_DARKER)]
# the widgets of the experimental intensity prior: hidden without INTENSITY_PRIOR_ENV and never persisted
INTENSITY_PRIOR_FIELDS = ("intensity_prior_checkbox", "intensity_prior_match_combobox",
                          "intensity_prior_slowdown_combobox", "intensity_prior_feature_weight_slider",
                          "intensity_prior_min_anchors_spinbox")


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

# a class for storing the original layer when creating a new labels layer as metadata
class SelectedLayerTracker:
    def __init__(self, viewer):
        self._prev_active = None
        viewer.layers.events.inserting.connect(self._store_selected)
        viewer.layers.events.inserted.connect(self._add_metadata)

    def _store_selected(self, event):
        if type(layer := event.source.selection.active) == Image:
            self._prev_active = layer

    def _add_metadata(self, event):
        if self._prev_active is None or type(event.value) is not Labels:
            return
        event.value.metadata["source_image_layer"] = self._prev_active
        self._prev_active = None


@magicclass(name="Minimal Contour", widget_type="scrollable", properties={"x_enabled": False})
class MinimalContourWidget(MagicTemplate):
    non_labels_layer_txt = field("Select a Labels layer.", widget_type="Label").with_options(visible=False)
    layer_has_no_source = field("The selected layer has no source image layer.", widget_type="Label").with_options(visible=False)

    @magicclass(layout="horizontal", name="source_image_container", labels=False)
    class SourceImageWidget(MagicTemplate):
        source_image_txt = field("Source: None", widget_type="Label")
        def change_source_image(self): ...
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
    # experimental, only shown when the INTENSITY_PRIOR_ENV environment variable is set
    intensity_prior_checkbox = field(bool, label="Follow anchor intensity", location=ImageFeaturesWidget).with_options(
        visible=False, tooltip="[Experimental] When checked, the contour follows pixels whose intensity is similar\n"
                               "to the intensity at the anchor points (and along the accepted contour)")
    intensity_prior_match_combobox = field(widget_type=ComboBox, label="Match", location=ImageFeaturesWidget, options={
        "choices": PRIOR_MATCH_CHOICES, "value": SIMILAR, "visible": False,
        "tooltip": "Which intensities count as matching the anchors:\n"
                   "Similar: close to it, darker or brighter\n"
                   "Similar or brighter: also anything brighter (e.g. along a membrane stain)\n"
                   "Similar or darker: also anything darker"})
    intensity_prior_slowdown_combobox = field(widget_type=ComboBox, label="Intensity strength", location=ImageFeaturesWidget, options={
        "choices": PRIOR_SLOWDOWN_CHOICES, "value": DEFAULT_PRIOR_SLOWDOWN, "visible": False,
        "tooltip": "How many times slower the contour moves on pixels whose intensity\n"
                   "differs most from the anchors' intensity"})
    intensity_prior_feature_weight_slider = field(float, label="Feature weight", widget_type="FloatSlider", location=ImageFeaturesWidget).with_options(
        min=0., max=1., value=DEFAULT_PRIOR_FEATURE_WEIGHT, visible=False,
        tooltip="How much the selected image feature still counts while following the intensity.\n"
                "0: intensity only; small values keep the contour tidy on noisy images")
    intensity_prior_min_anchors_spinbox = field(int, label="Prior from anchor", location=ImageFeaturesWidget).with_options(
        min=1, max=10, value=1, visible=False,
        tooltip="Number of placed anchor points after which the prior is used.\n"
                "With 1 the intensity at the first anchor is the reference;\n"
                "more anchors give a more reliable reference.")


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
        self._viewer: napari.Viewer = unwrap_viewer(viewer)
        self._labels_layer: Labels = None
        self.apply_contrast_limits = True
        # multiscale image layers that were already reported as unsupported
        self._warned_multiscale = weakref.WeakSet()
        self.point_triangle = np.zeros((3, 2), dtype=np.float64) - 1  # start point, current position, end point
        self.remove_last_anchor = False
        self.last_added_with_shift = None
        self.last_segment_length = None
        self.prev_n_anchor_points = 0
        self.prev_labels_layer = self.labels_layer
        self.feature_inverted = False
        self.prev_timer_id = None
        # gradients of the displayed plane, computed when needed (see _get_gradients)
        self._plane_features = PlaneFeatureCache()
        # features are computed in a worker thread while scrolling; only the newest request is applied
        self._feature_generation = 0
        self._feature_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mc-features")
        self._feature_bridge = _FeatureBridge()
        self._feature_bridge.ready.connect(self._apply_features)
        # the part of the plane that is handed to the calculator is moved when the search area leaves it
        self._roi_hint = None
        request_roi = weakref.WeakMethod(self._on_roi_request)

        def roi_requester(search_box):
            method = request_roi()
            if method is not None:
                method(search_box)
        VispyMinimalContourOverlay.roi_requester = roi_requester
        # per image layer id: bumped when the layer's data changes, so that its cached planes are never reused
        self._data_versions = {}
        # image layers whose data event is already connected
        self._data_watch = weakref.WeakSet()
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
        self.selected_layer_tracker = None

    def __post_init__(self):
        if self._viewer is not None:
            self._initialize(self._viewer)
        VispyMinimalContourOverlay.mc_calculator.set_param(self.param)
        self.local_intensity_correction_checkbox.changed.connect(VispyMinimalContourOverlay.mc_calculator.set_use_local_maximum)
        if os.environ.get(INTENSITY_PRIOR_ENV, "") not in ("", "0"):
            for name in INTENSITY_PRIOR_FIELDS:
                getattr(self, name).visible = True
        for widget, setter in [
            (self.intensity_prior_match_combobox, VispyMinimalContourOverlay.set_intensity_prior_match),
            (self.intensity_prior_slowdown_combobox, VispyMinimalContourOverlay.set_intensity_prior_max_slowdown),
            (self.intensity_prior_feature_weight_slider, VispyMinimalContourOverlay.set_intensity_prior_feature_weight),
            (self.intensity_prior_min_anchors_spinbox, VispyMinimalContourOverlay.set_intensity_prior_min_anchors),
            (self.intensity_prior_checkbox, VispyMinimalContourOverlay.set_use_intensity_prior),
        ]:
            setter(widget.value)
            widget.changed.connect(setter)

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
        self.blur_sigma_slider.native.children()[0].sliderReleased.connect(self._set_image)
        self._on_label_change()
        self._on_selected_label_change()

    def __magicclass_serialize__(self):
        d = serialize(self)
        if "labels_layer_combobox" in d:
            del d["labels_layer_combobox"]
        if "source_image_container" in d:
            del d["source_image_container"]
        # the experimental intensity prior always starts switched off
        for container in (d, d.get("Image Features", {})):
            for key in INTENSITY_PRIOR_FIELDS:
                container.pop(key, None)
        return d

    @property
    def viewer(self):
        return self._viewer

    @property
    def image_layer(self):
        return self.labels_layer.metadata.get("source_image_layer", None) if self.labels_layer is not None else None

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
    def is_mc_enabled(self):
        """Whether the minimal contour tool is on for the active labels layer; the image is only processed then."""
        labels_layer = self.labels_layer
        if labels_layer is None:
            return False
        overlay = labels_layer._overlays.get("minimal_contour")
        return overlay is not None and overlay.enabled

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
        done = Signal(object)
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

    def _on_image_changed(self, *args):
        self._store_orig_image(*args)
        self.feature_editor.run_button.setEnabled(self.image_layer is not None)
        self.feature_editor.run_button.setToolTip("Open an image first" if self.image_layer is None else "")
        self._set_image("_on_image_changed")

    def _on_label_change(self, _=None):
        """Bind the Label Options section to the current labels layer (and unbind it from the previous one)."""
        labels_layer = self.labels_layer
        selected_label_spinbox = self.selected_label_spinbox
        if labels_layer is not None:
            dtype_lims = get_dtype_limits(get_dtype(labels_layer))
        else:
            dtype_lims = 0, np.iinfo(int).max
        selected_label_spinbox.min, selected_label_spinbox.max = dtype_lims
        self.color_box.layer = labels_layer
        prev_labels_layer = self.prev_labels_layer
        if prev_labels_layer is not None and prev_labels_layer is not labels_layer:
            prev_labels_layer.events.selected_label.disconnect(self._on_selected_label_change)
            if self._on_mouse_wheel in prev_labels_layer.mouse_wheel_callbacks:
                prev_labels_layer.mouse_wheel_callbacks.remove(self._on_mouse_wheel)
        if labels_layer is not None and labels_layer is not prev_labels_layer:
            labels_layer.events.selected_label.connect(self._on_selected_label_change)
            if self._on_mouse_wheel not in labels_layer.mouse_wheel_callbacks:
                labels_layer.mouse_wheel_callbacks.append(self._on_mouse_wheel)
        self.prev_labels_layer = labels_layer
        if labels_layer is not None:
            self._on_selected_label_change()
        elif selected_label_spinbox.value != 0:
            selected_label_spinbox.value = 0

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
        if self.viewer is None:
            return
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
                slice_indices = layer_slice_indices(im_layer)
                data = im_layer.data[0] if im_layer.multiscale else im_layer.data
                # np.asarray: the layer data may be lazy (e.g. dask)
                img = np.asarray(data[slice_indices]).astype(float)
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
        # some platforms turn Shift/Alt+wheel into horizontal scrolling, so take whichever axis moved
        delta = (np.sign(event.delta)).astype(int)
        step = delta[1] if delta[1] != 0 else delta[0]
        modifiers = event.modifiers
        if "Shift" in modifiers:
            labels_layer.selected_label = max(0, labels_layer.selected_label + step)
        elif "Alt" in modifiers:
            diff = step
            diff *= min(max(1, labels_layer.brush_size//10), 5)
            labels_layer.brush_size = max(1, labels_layer.brush_size + diff)

    def _blur_image(self, img, sigma=None):
        return gaussian(
            img,
            sigma=self.blur_sigma if sigma is None else sigma,
            preserve_range=True,
            channel_axis=-1 if img.ndim == 3 else None,
            truncate=2.
        )

    def _store_orig_image(self, *_):
        """Make releasing the source image's contrast limit slider recompute the features.

        The image layer's data is never copied or modified: blurring is applied to the plane given to the calculator.
        """
        image_layer: Image = self.image_layer
        qt_viewer = viewer_qt_viewer(self.viewer)
        if image_layer in qt_viewer.controls.widgets:
            controls = qt_viewer.controls.widgets[image_layer]
            for child in controls.children():
                if type(child) == QDoubleRangeSlider:
                    try:
                        child.sliderReleased.connect(self._set_image, Qt.UniqueConnection)
                    except TypeError:
                        pass
                    break

    def _set_image(self, event=None):
        self._update_features()

    def _update_features(self, background=False):
        """Recompute the features of the displayed plane.

        With ``background`` the numpy work runs in a worker thread and only handing the result to the calculator is
        done in the GUI thread, so the viewer stays responsive (used while scrolling through slices).
        """
        self._feature_generation += 1  # drops the result of any computation still running
        image_layer: Image = self.image_layer
        if image_layer is None:
            self._img = None
            self.feature_editor.variables["image"] = None
            VispyMinimalContourOverlay.set_calculator_feature(None)
            VispyMinimalContourOverlay.set_prior_image(None)
            # self.feature_editor.features = None #TODO check
            return
        if self.viewer is None:
            return
        if not self.is_mc_enabled:
            # nothing to compute while the tool is off (e.g. only painting); enabling it calls this again
            VispyMinimalContourOverlay.set_calculator_feature(None)
            VispyMinimalContourOverlay.set_prior_image(None)
            return
        if image_layer.multiscale:
            # without features the overlay ignores every mouse event, so the tool stays inactive
            if image_layer not in self._warned_multiscale:
                self._warned_multiscale.add(image_layer)
                warnings.warn("Minimal contour does not support multiscale images (layer '%s')" % image_layer.name)
            VispyMinimalContourOverlay.set_calculator_feature(None)
            VispyMinimalContourOverlay.set_prior_image(None)
            return
        if self.viewer.dims.ndisplay == 3:
            # self.anchor_points.editable = False
            return
        if not image_layer.visible:
            image_layer.set_view_slice()
        image_layer._slice_dims(self.viewer.dims)
        self.autoincrease_label_id = (image_layer.ndim == 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plane = image_layer._data_view
        if self.used_feature == CUSTOM:
            image = plane.astype(float)
            if image.ndim == 2:
                image = np.concatenate([image[..., np.newaxis]] * 3, axis=2)
            elif image.shape[-1] == 4:
                image = image[..., :3]
            min_, max_ = image_layer.contrast_limits if self.apply_contrast_limits else (image.min(), image.max())
            image = np.clip((image - min_) / (max_ - min_), 0, 1)
            if self.feature_inverted:
                image = 1 - image
            self._img = image
            # The plane is blurred once, with the slider's sigma. Before, the plane was also blurred inside the
            # image layer first, so the effective sigma was sqrt(2) * sigma.
            if self.is_blurring_enabled:
                image = self._blur_image(image).astype(float)
            VispyMinimalContourOverlay.set_prior_image(1 - image if self.feature_inverted else image)
            self.feature_editor.variables["image"] = image
            self.feature_editor.Run()
            return
        request = self._gradient_request(image_layer)
        if request is None:
            return
        job = dict(
            plane=plane,
            gradients=request,
            roi=self._choose_roi(image_layer, plane.shape[:2]),
            limits=tuple(image_layer.contrast_limits) if self.apply_contrast_limits else None,
            inverted=self.feature_inverted,
            sigma=self.blur_sigma if self.is_blurring_enabled else None,
            feature=self.used_feature,
        )
        self._feature_generation += 1
        generation = self._feature_generation
        if not background:
            self._apply_features(generation, self._compute_features(job))
            return

        def work():
            if generation != self._feature_generation:  # a newer plane was requested meanwhile
                return
            try:
                result = self._compute_features(job)
            except Exception as e:
                warnings.warn("Computing the minimal contour features failed: %r" % e)
                return
            self._feature_bridge.ready.emit(generation, result)
        self._feature_executor.submit(work)

    def _compute_features(self, job):
        """Features, x gradient, y gradient (the latter two may be None), the ROI and the intensity image of a plane.

        Without a ROI the ROI item is None. With one, everything is computed on the ROI plus the pixels around it that
        gradients and blur need, and cropped back to the ROI: inside it the result equals the whole plane's, except for
        the statistics the features are normalised with when no contrast limits are used, or for the symmetric
        gradient (its magnitude is scaled to 0..1): those are the ROI's own, which is what makes the cost independent
        of the plane size. The ROI item is (Box, shape of the whole plane). The intensity image (normalised and blurred
        like the features, same window, but not inverted) is what the intensity prior samples. Thread safe.
        """
        plane = job["plane"]
        sigma = job["sigma"]
        roi = job["roi"]
        key, _, rgb = job["gradients"]
        image = np.asarray(plane)
        plane_shape = image.shape[:2]
        inner = None
        if roi is not None:
            window = mc_roi.padded(roi, plane_shape, mc_roi.feature_padding(rgb, sigma))
            inner = (slice(roi.y0 - window.y0, roi.y1 - window.y0), slice(roi.x0 - window.x0, roi.x1 - window.x0))
            image = image[window.slices]
            plane = plane[window.slices]
            key = key + (tuple(window),)
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        image = image.astype(float)
        # a gray plane stays 2D here: its 3 equal channels would only triple the cost of the steps below
        if job["limits"] is not None:
            min_, max_ = job["limits"]
        else:
            reference = image if inner is None else image[inner]
            min_, max_ = reference.min(), reference.max()
        image = np.clip((image - min_) / (max_ - min_), 0, 1)
        if job["inverted"]:
            image = 1 - image
        if sigma is not None:
            image = self._blur_image(image, sigma).astype(float)
        grad_x, grad_y = to_float64_pair(self._plane_features.get(key, lambda: plane, rgb))
        if sigma is not None:
            grad_x = self._blur_image(grad_x, sigma)
            grad_y = self._blur_image(grad_y, sigma)
        if inner is not None:
            image, grad_x, grad_y = (np.ascontiguousarray(a[inner]) for a in (image, grad_x, grad_y))
            roi = (roi, tuple(plane_shape))
        # the prior compares intensities as they are displayed, also for the inverted (low intensity) feature
        intensity = 1 - image if job["inverted"] else image
        if job["feature"] == SYM_GRADIENT:
            grad_x, grad_y = grad_x / (max_ - min_), grad_y / (max_ - min_)
            grad_magnitude = np.linalg.norm([grad_x, grad_y], axis=0)
            min_, max_ = grad_magnitude.min(), grad_magnitude.max()
            return (grad_magnitude - min_) / (max_ - min_), None, None, roi, intensity
        if image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
        return image, grad_x, grad_y, roi, intensity

    def _apply_features(self, generation, result):
        """Hand features to the calculator (GUI thread), unless a newer plane was requested or the tool was left."""
        if generation != self._feature_generation or not self.is_mc_enabled or self.image_layer is None:
            return
        features, grad_x, grad_y, roi, intensity = result
        self._img = features if grad_x is not None else None
        # before the calculator gets the features: that recomputes the prior's speed modulation
        VispyMinimalContourOverlay.set_prior_image(intensity)
        if roi is None:
            VispyMinimalContourOverlay.set_calculator_feature(features, grad_x, grad_y)
        else:
            box, plane_shape = roi
            VispyMinimalContourOverlay.set_calculator_feature(
                features, grad_x, grad_y, roi_origin=box.origin, plane_shape=plane_shape)

        # self.anchor_points.translate = image_layer.translate
        # self.from_e_points_layer.translate = image_layer.translate
        # self.to_s_points_layer.translate = image_layer.translate
        # self.output.translate = image_layer.translate

    def _choose_roi(self, image_layer: Image, plane_shape):
        """The part (a Box) of the displayed plane to give the calculator, or None for the whole plane.

        Sources, in this order: the bounding box of the object selected in the object list (only while the contour
        runs on that very image: a cropped copy is small already), and, for planes with more than
        ``LARGE_PLANE_PIXELS`` pixels, a window around the anchors. ``_roi_hint`` is a search area that has to be
        inside the result (set while the overlay asks for a new window). Nothing is windowed for the custom
        features, whose script is given the whole plane.
        """
        if self.used_feature == CUSTOM or len(plane_shape) != 2:
            return None
        hint = self._roi_hint
        try:
            dims_displayed = tuple(layer_dims_displayed(image_layer))
            selected = selected_object(self.viewer)
            if selected is not None and selected[3] is image_layer:
                box = mc_roi.object_box(selected[1], selected[2], dims_displayed, plane_shape)
                if box is not None and hint is not None:
                    box = mc_roi.union(box, mc_roi.expand(hint, plane_shape))
                return mc_roi.use_roi(box, plane_shape)
            if plane_shape[0] * plane_shape[1] <= mc_roi.LARGE_PLANE_PIXELS:
                return None
            if hint is not None:
                points = [(hint.y0, hint.x0), (hint.y1 - 1, hint.x1 - 1)]
            else:
                points = self._window_center(image_layer, dims_displayed, plane_shape)
            return mc_roi.use_roi(mc_roi.window_around(points, plane_shape), plane_shape)
        except Exception as e:  # the ROI is only an optimisation: without one the whole plane is used
            warnings.warn("Could not derive the minimal contour ROI, using the whole plane: %r" % e)
            return None

    def _window_center(self, image_layer: Image, dims_displayed, plane_shape):
        """Where to put the window of a big plane (rows, columns): at the anchors, else at the middle of the view."""
        overlay = self.labels_layer._overlays.get("minimal_contour") if self.labels_layer is not None else None
        anchors = None
        if overlay is not None and len(overlay.anchor_points):
            anchors = np.asarray(overlay.anchor_points, dtype=float)
        if anchors is not None and anchors.ndim == 2 and anchors.shape[1] > max(dims_displayed):
            return anchors[:, list(dims_displayed)]
        try:
            world = list(self.viewer.dims.point)
            center = viewer_camera(self.viewer).center
            world[dims_displayed[0]], world[dims_displayed[1]] = center[-2], center[-1]
            data = np.asarray(image_layer.world_to_data(world))[list(dims_displayed)]
            if np.all(np.isfinite(data)):
                return [data]
        except Exception:
            pass
        return [(plane_shape[0] // 2, plane_shape[1] // 2)]

    def _on_roi_request(self, search_box):
        """The overlay needs to search ``search_box``, which is outside of the part of the plane the calculator has."""
        if not self.is_mc_enabled:
            return
        self._roi_hint = search_box
        try:
            self._update_features(background=False)
        finally:
            self._roi_hint = None

    def _get_gradients(self, image_layer: Image):
        request = self._gradient_request(image_layer)
        if request is None:
            return None, None
        key, plane_provider, rgb = request
        return self._plane_features.get(key, plane_provider, rgb)

    def _gradient_request(self, image_layer: Image):
        """(cache key, plane provider, rgb) for the gradients of the displayed plane, or None if there are none.

        The x and y derivative of the plane displayed for ``image_layer`` (float32, read-only), or (None, None).

        The plane is taken as displayed (``_data_view``, so x and y follow the display order) and its gradients are
        cached per (layer, displayed dims, slice indices), so only the planes that are visited are processed.
        """
        if layer_ndisplay(image_layer) == 3 or image_layer not in self.viewer.layers:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dims_displayed = tuple(layer_dims_displayed(image_layer))
            dims_not_displayed = tuple(layer_dims_not_displayed(image_layer))
            slice_indices = layer_slice_indices(image_layer)
        idx = tuple(slice_indices[i] for i in dims_not_displayed)
        shape = image_layer.data.shape
        if any(i < 0 or i >= shape[dim] for dim, i in zip(dims_not_displayed, idx)):
            return None
        layer_id = id(image_layer)
        if image_layer not in self._data_watch:
            self._data_watch.add(image_layer)
            image_layer.events.data.connect(lambda *_: self._on_image_data_change(layer_id))
        key = (layer_id, dims_displayed, idx, self._data_versions.get(layer_id, 0))

        def plane():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return image_layer._data_view
        return key, plane, image_layer.rgb

    def _on_image_data_change(self, layer_id):
        self._data_versions[layer_id] = self._data_versions.get(layer_id, 0) + 1
        self._plane_features.invalidate(layer_id)

    def _on_layer_removed(self, layer):
        layer_id = id(layer)
        self._plane_features.invalidate(layer_id)
        self._data_versions.pop(layer_id, None)
        self._data_watch.discard(layer)

    @delay_function(delay=.5, leading=False)
    def _delayed_set_image(self, *args):
        self._update_features(background=True)

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
        viewer_qt_viewer(self.viewer).canvas.native.setFocus()

    def _hide_widgets(self):
        self.ImageFeaturesWidget.visible = False
        self.LabelOptionsWidget.visible = False
        self.ContourWidget.visible = False
        self.BlurWidget.visible = False
        self.non_labels_layer_txt.visible = False
        self.layer_has_no_source.visible = False
        self.SourceImageWidget.visible = False

    def _show_no_labels_layer_selected_txt(self):
        self._hide_widgets()
        self.non_labels_layer_txt.visible = True

    def _show_layer_has_no_source_txt(self):
        self._hide_widgets()
        self.layer_has_no_source.visible = True
        self.SourceImageWidget.visible = True

    def _show_base_widgets(self):
        self._hide_widgets()
        self.ImageFeaturesWidget.visible = True
        self.LabelOptionsWidget.visible = True
        self.ContourWidget.visible = True
        self.BlurWidget.visible = True
        self.SourceImageWidget.visible = True
        image_layer = self.image_layer
        self.SourceImageWidget.source_image_txt.value = "Source: %s" % (image_layer.name if image_layer is not None else "None")

    def _select_source_image_dialog(self, labels_layer):
        """Show a dialog to select/change the source image layer for labels_layer.
        Returns True if a source was successfully assigned, False otherwise."""
        image_layers = list(map(lambda layer: layer.name, filter(lambda layer: isinstance(layer, Image), self.viewer.layers)))
        if len(image_layers) == 0:
            return False
        dialog = QDialog(self.native)
        dialog.setWindowTitle("Select Source Image")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Please select the source image layer for '%s'." % labels_layer.name))
        combo_box = QComboBox(dialog)
        combo_box.addItems(image_layers)
        # pre-select the current source image if set
        current = labels_layer.metadata.get("source_image_layer")
        if current is not None and current.name in image_layers:
            combo_box.setCurrentIndex(image_layers.index(current.name))
        layout.addWidget(combo_box)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        if dialog.exec() != QDialog.Accepted:
            return False
        source_layer = self.viewer.layers[image_layers[combo_box.currentIndex()]]
        if source_layer.ndim != labels_layer.ndim or source_layer.data.shape[:source_layer.ndim] != labels_layer.data.shape:
            if QMessageBox.question(self.native, "Resizing required",
                    f"The sizes of '{labels_layer.name}' and '{source_layer.name}' do not match.\n"
                    f"Do you want to resize '{labels_layer.name}'? This will reset its current content."
                    ) != QMessageBox.StandardButton.Yes:
                return False
            name = labels_layer.name
            dtype = labels_layer.data.dtype
            with self.viewer.layers.selection.events.active.blocker(self._on_active_layer_changed):
                self.viewer.layers.remove(labels_layer)
            new_labels = self.viewer.add_labels(
                np.zeros(source_layer.data.shape[:source_layer.ndim], dtype=dtype), name=name)
            new_labels.metadata["source_image_layer"] = source_layer
            self.viewer.layers.selection.select_only(new_labels)
            return True
        labels_layer.metadata["source_image_layer"] = source_layer
        return True

    @set_design(location=SourceImageWidget, text="Change")
    def change_source_image(self):
        """Change the source image layer for the currently active labels layer."""
        if self.labels_layer is None:
            return
        if self._select_source_image_dialog(self.labels_layer):
            self._on_image_changed()
            self.SourceImageWidget.source_image_txt.value = "Source: %s" % (
                self.image_layer.name if self.image_layer is not None else "None")

    def _on_active_layer_changed(self, *_):
        if self.viewer is None:
            return
        active_layer = self.viewer.layers.selection.active
        self._labels_layer = active_layer if isinstance(active_layer, Labels) else None
        self._on_label_change()
        if not self._labels_layer:
            self._show_no_labels_layer_selected_txt()
            return
        if "source_image_layer" not in self._labels_layer.metadata:
            image_layers = list(map(lambda layer: layer.name, filter(lambda layer: isinstance(layer, Image), self.viewer.layers)))
            if len(image_layers) == 0:
                self._show_layer_has_no_source_txt()
                return
            assigned = self._select_source_image_dialog(self._labels_layer)
            if not assigned:
                self._show_layer_has_no_source_txt()
                return
            # if _select_source_image_dialog triggered a layer replacement it calls select_only
            # which re-triggers this callback — bail out early to avoid double-processing
            if self._labels_layer not in self.viewer.layers:
                return
        if "minimal_contour" not in self._labels_layer._overlays:
            bind_all(self._labels_layer, LABELS, {"straight_lines": self._on_ctrl_pressed})
            self._labels_layer._overlays.update({"minimal_contour": MinimalContourOverlay()})
            self._labels_layer._overlays["minimal_contour"].visible=True
            self._labels_layer._overlays["minimal_contour"].events.enabled.connect(self._on_mc_enabled_change)
            labels_control: QtLabelsControls = viewer_qt_viewer(self.viewer).controls.widgets[self._labels_layer]
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
                self._labels_layer.mode = Mode.PAN_ZOOM
                self._enable_mc_mode()
            bind_all(self._labels_layer, LABELS, {"minimal_contour_mode": switch_to_mc_mode})
        self.labels_layer._overlays["minimal_contour"].contour_smoothness = self.contour_smoothness if self.is_contour_smoothing_enabled else 1.
        self._on_image_changed()
        self._show_base_widgets()

    def _on_ctrl_pressed(self, _):
        if self.labels_layer is None:
            return
        self.labels_layer._overlays["minimal_contour"].use_straight_lines = True
        yield
        self.labels_layer._overlays["minimal_contour"].use_straight_lines = False


    def _set_viewer_callbacks(self):
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

        def change_layer_callback(num):
            def change_layer(_):
                if num - 1 < len(self.viewer.layers):
                    self.viewer.layers.selection.select_only(self.viewer.layers[num - 1])
                else:
                    warnings.warn("%d is out of range (number of layers: %d)" % (num, len(self.viewer.layers)))

            return change_layer

        bind_all(self.viewer, VIEWER, {"jump_to_layer": lambda key: change_layer_callback(int(key[-1]))})
        self.viewer.dims.events.current_step.connect(self._delayed_set_image, position="last")

        self.viewer.dims.events.ndisplay.connect(self._set_image, position="last")
        self.viewer.dims.events.order.connect(self._set_image, position="last")
        self.viewer.layers.events.removed.connect(lambda e: self._on_layer_removed(e.value))
        # key and wheel events go to the focused widget (usually the canvas), not to the main window
        viewer_main_window(self.viewer).installEventFilter(self.event_filter)
        viewer_qt_viewer(self.viewer).canvas.native.installEventFilter(self.event_filter)
        viewer_main_window(self.viewer).setMouseTracking(True)

    def _initialize(self, viewer: napari.Viewer):
        viewer = self._viewer = unwrap_viewer(viewer)
        self._initialize_helper_layers()
        # feature files of older versions that were left behind (this version writes none)
        clean_legacy_tmp()
        self._selected_layer_tracker = SelectedLayerTracker(self.viewer)
        self._set_viewer_callbacks()
        self._store_orig_image()
        self._update_label_tooltip()
        self._set_image()
        action_manager.register_action("napari-nD-annotator:activate_labels_mc_mode", self._enable_mc_mode,
                                       "We're switching to MC mode", None)

    def _check_mc_button(self):
        """Tick the minimal contour button, taking the check back from pan/zoom.

        The button has no mode of its own, so it borrows PAN_ZOOM. Clicking it
        while another tool is active is a real mode transition, and napari
        answers the mode event by ticking its own pan/zoom button, which steals
        our check because the two share an exclusive button group. The clicked
        signal that brings us here fires after that cascade has unwound, so it
        is late enough to claim the check back.
        """
        if self.viewer is None or self.labels_layer is None:
            return
        controls = viewer_qt_viewer(self.viewer).controls.widgets.get(self.labels_layer)
        mc_button = getattr(controls, "mc_button", None)
        if mc_button is None or mc_button.isChecked():
            return
        # Blocked so _set_mode doesn't run again; the group still unchecks pan/zoom.
        mc_button.blockSignals(True)
        mc_button.setChecked(True)
        mc_button.blockSignals(False)

    def _enable_mc_mode(self, *args, **kwargs):
        if self.labels_layer is not None:
            self._check_mc_button()
            self.labels_layer._overlays["minimal_contour"].enabled = True

    def _disable_mc_mode(self, state):
        if state and self.labels_layer is not None:
            self.labels_layer._overlays["minimal_contour"].enabled = False

    def _on_mc_enabled_change(self, *_):
        """Prepare the image features when the tool is turned on (by any means), drop them when it is turned off.

        _set_image looks at the active labels layer, so a change on another layer only refreshes the current state.
        """
        self._set_image("_on_mc_enabled_change")
