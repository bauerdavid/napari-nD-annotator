import warnings

from magicgui.widgets import PushButton
from time import sleep
from copy import deepcopy
from collections.abc import Iterable
from packaging import version
from threading import Thread
import numpy as np
import itertools
import colorsys
from random import shuffle, seed

#image processing modules
import cv2
import skimage.measure, skimage.morphology
from scipy.ndimage import gaussian_filter, zoom

#UI
from magicclass import magicclass, MagicTemplate, field, vfield, set_design
from magicclass.serialize import serialize
from magicclass.widgets import FreeWidget

from qtpy.QtCore import Signal, Qt, QObject, QThread
from qtpy.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QCheckBox,
    QPushButton,
    QFrame,
    QWidget,
    QApplication,
    QMessageBox
)

import napari
from napari.layers import Image, Labels, Points, Shapes, Layer
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.notifications import show_info
from napari.layers._source import layer_source


from ._utils import WidgetWithLayerList, ProgressWidget, QDoubleSlider, QSymmetricDoubleRangeSlider, ScriptExecuteWidget
from ._utils.callbacks import keep_layer_on_top
from .._helper_functions import layer_dims_displayed
from ._utils.collapsible_widget import CollapsibleContainerGroup, correct_container_size
from .minimal_contour_widget import MinimalContourWidget, delay_function
from .._napari_version import NAPARI_VERSION


try:
    from napari_bbox import BoundingBoxLayer
except ImportError:
    BoundingBoxLayer = None

try:
    import minimal_surface
except ImportError:
    minimal_surface = None


GRADIENT = "Gradient"
CUSTOM = "Custom"
FEATURE_KEYS = [GRADIENT, CUSTOM]
# FEATURES = {
#     ASSYM_GRADIENT: GRADIENT_BASED,
#     SYM_GRADIENT: INTENSITY_BASED,
#     HIGH_INTENSITY: INTENSITY_BASED,
#     LOW_INTENSITY: INTENSITY_BASED,
#     CUSTOM: INTENSITY_BASED
# }

CAD_BLURRING = "Edge preserving" #"Curvature Anisotropic Diffusion"
GAUSSIAN_BLURRING = "Gaussian"

def pts_2_bb(p1, p2, image_size, scale=1.):
    center = (p1 + p2) / 2
    size = np.divide(np.sqrt((np.multiply(p1 - p2, scale) ** 2).sum()), scale) + 10
    bb = np.asarray(
        np.where(list(itertools.product((False, True), repeat=3)), center + size / 2, center - size / 2))
    bb = np.clip(bb, 0, np.asarray(image_size) - 1)
    return bb


def get_bb_corners(bb):
    return np.concatenate([bb.min(0, keepdims=True), bb.max(0, keepdims=True)])


def bb_2_slice(bb):
    bb_corners = get_bb_corners(bb).round().astype(int)
    return tuple(slice(bb_corners[0, i], bb_corners[1, i]) for i in range(3))


def color_to_hex_string(rgb_tuple):
    return '#%.2X%.2X%.2X' % rgb_tuple


def generate_label_colors(n):
    rgb_tuples = [colorsys.hsv_to_rgb(x * 1.0 / n, 1., 1.) for x in range(n)]
    rgb_tuples = [(int(rgb_tuples[i][0] * 255), int(rgb_tuples[i][1] * 255), int(rgb_tuples[i][2] * 255)) for i in
                  range(n)]
    seed(0)
    shuffle(rgb_tuples)
    return rgb_tuples


def generate_sphere_mask(shape):
    ndim = len(shape)
    r = max(shape) // 2
    mask = skimage.morphology.ball(r)
    shape_diffs = tuple(shape[i] - mask.shape[i] for i in range(ndim))
    crop_idx = tuple(
        slice(max(0, round(-shape_diffs[i] / 2 + 0.1)), mask.shape[i] + min(0, round(shape_diffs[i] / 2 + 0.1))) for
        i in range(ndim))
    mask = mask[crop_idx]
    mask = np.pad(mask, [(max(0, round(shape_diffs[i] / 2)), -min(0, round(-shape_diffs[i] / 2 + 0.1))) for i in
                         range(ndim)])
    assert mask.shape == shape
    return mask


class ColorPairsCallback:
    def __init__(self, n_colors=50):
        self.prev_selection = set()
        self.prev_len = 0
        self.prev_data = None
        color_cycle = list(
            map(lambda color_tuple: color_to_hex_string(color_tuple), generate_label_colors(n_colors)))
        self.color_cycle = itertools.cycle(color_cycle)
        self._last_color = None

    def __call__(self, event):
        if event.type != "data" or getattr(event, "action", "add") not in ["add", "added"]:
            return
        points_layer = event.source
        if len(points_layer.selected_data) > 1:
            points_layer.selected_data = self.prev_selection
            points_layer.refresh()
            return
        self.prev_selection = points_layer.selected_data.copy()
        if len(points_layer.data) > self.prev_len:
            if len(points_layer.data) % 2 == 1:
                self._last_color = next(self.color_cycle)
                # points_layer.current_face_color = new_color
            points_layer.face_color[-1] = transform_color(self._last_color)
            self.prev_data = points_layer.data.copy()
        elif len(points_layer.data):
            removed_idx = np.squeeze(np.argwhere(
                np.all(np.any(~np.equal(self.prev_data[np.newaxis], points_layer.data[:, np.newaxis]), -1),
                       axis=0)))
            if removed_idx.ndim == 0:
                removed_idx = int(removed_idx)
                points_layer.selected_data.clear()
                if removed_idx % 2 == 0 and removed_idx < len(points_layer.data):
                    points_layer.selected_data.add(removed_idx)
                elif removed_idx % 2 == 1:
                    points_layer.selected_data.add(removed_idx - 1)
                points_layer.remove_selected()
        points_layer.refresh()
        self.prev_data = points_layer.data
        self.prev_len = len(points_layer.data)


class EstimationWorker(QObject):
    image_data_received = Signal(str, "PyQt_PyObject", "PyQt_PyObject")
    mask_data_received = Signal("PyQt_PyObject", "PyQt_PyObject")
    remove_layer = Signal(str)
    layer_invalidated = Signal(str)
    all_done = Signal()
    annotation_needed_signal = Signal("PyQt_PyObject", "PyQt_PyObject")
    slice_annotations_done = Signal(int)
    object_annotated = Signal(int)
    MANUAL_ANNOTATION = "Manual"
    THRESHOLD_ANNOTATION = "Threshold"
    MINIMAL_CONTOUR_ANNOTATION = "Minimal contour"
    ANNOTATION_METHODS = [MANUAL_ANNOTATION, THRESHOLD_ANNOTATION, MINIMAL_CONTOUR_ANNOTATION]

    def __init__(self, viewer, minimal_contour_widget: MinimalContourWidget, minimal_surface_widget, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.minimal_contour_widget = minimal_contour_widget
        self.minimal_surface_widget = minimal_surface_widget
        self.estimators = None
        self.bboxes = None
        self.imgs = None
        self.phis = None
        self.pt_pairs = None
        self.stop_requested = False
        self.center_mat = None
        self.rotation_matrix = None
        self.translation_matrix = None
        self.phi = None
        self.image = None
        self.points = None
        self.use_correction = None
        self.beta = None
        self.alpha = None
        self.n_iter = None
        self.use_meeting_plane_points = None
        self.done_pressed = False
        self.blur_func = None
        self.init_slice_annotation_fun = None
        self.finish_slice_annotation_fun = None
        self.bb_layer = None
        self.slice_image_layer = None
        self.slice_labels_layer = None
        self.slice_annotation = None
        self.manual_annotation_done = False
        self.z_scale = None
        self.use_gradient = None

        self.prev_ndisplay = None
        self.prev_camera_center = None
        self.prev_camera_zoom = None
        self.prev_camera_angles = None

        #create annotation dialogs
        self.manual_annotation_dialog = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Annotate image, then click 'Done'"))
        annotation_done_button = QPushButton("Done")
        layout.addWidget(annotation_done_button)
        def on_done_pressed():
            self.done_pressed = True
        annotation_done_button.clicked.connect(self.finish_slice_annotation)
        annotation_done_button.clicked.connect(on_done_pressed)
        self.manual_annotation_dialog.setLayout(layout)
        self.manual_annotation_dialog.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Dialog)
        self.manual_annotation_dialog.setAttribute(Qt.WA_ShowWithoutActivating)

        self.threshold_annotation_dialog = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Threshold:"))
        threshold_slider = QDoubleSlider(Qt.Horizontal)
        threshold_slider.valueChanged.connect(self.threshold_value_changed)
        layout.addWidget(threshold_slider)
        annotation_done_button = QPushButton("Done")
        layout.addWidget(annotation_done_button)
        annotation_done_button.clicked.connect(self.finish_slice_annotation)
        annotation_done_button.clicked.connect(on_done_pressed)
        self.threshold_annotation_dialog.setLayout(layout)
        self.annotation_needed_signal.connect(self.begin_slice_annotation)
        self.slice_annotation_methods = {EstimationWorker.MANUAL_ANNOTATION: self.manual_annotation, EstimationWorker.THRESHOLD_ANNOTATION: self.threshold_annotation, EstimationWorker.MINIMAL_CONTOUR_ANNOTATION: self.minimal_contour_annotation}
        self.finish_annotation_methods = {EstimationWorker.MANUAL_ANNOTATION: self.finish_manual_annotation, EstimationWorker.THRESHOLD_ANNOTATION: self.finish_threshold_annotation, EstimationWorker.MINIMAL_CONTOUR_ANNOTATION: self.finish_minimal_contour_annotation}

    def begin_slice_annotation(self, image_slice, distance_map_slice):
        self.slice_image_layer = self.viewer.add_image(image_slice)
        self.slice_labels_layer = self.viewer.add_labels(np.zeros_like(image_slice, dtype=int))
        self.prev_ndisplay = self.viewer.dims.ndisplay
        self.prev_camera_center = self.viewer.camera.center
        self.prev_camera_zoom = self.viewer.camera.zoom
        self.prev_camera_angles = self.viewer.camera.angles
        self.viewer.dims.ndisplay = 2
        visible_extent = self.slice_labels_layer.extent.world[:, layer_dims_displayed(self.slice_labels_layer)]
        self.viewer.camera.center = visible_extent.mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.viewer.camera.zoom = np.min(np.divide(self.viewer._canvas_size, visible_extent.max(0)-visible_extent.min(0)))*0.95
        self.slice_labels_layer.brush_size = 1
        self.slice_labels_layer.mode = "paint"
        self.minimal_surface_widget.call_button.enabled = True
        self.init_slice_annotation_fun()

    def finish_slice_annotation(self):
        self.finish_slice_annotation_fun()
        self.minimal_surface_widget.call_button.enabled = False
        if self.slice_labels_layer is not None:
            self.slice_annotation = self.slice_labels_layer.data > 0
        if self.slice_image_layer in self.viewer.layers:
            self.viewer.layers.remove(self.slice_image_layer)
        if self.slice_labels_layer in self.viewer.layers:
            self.viewer.layers.remove(self.slice_labels_layer)
        if self.prev_ndisplay is not None:
            self.viewer.dims.ndisplay = self.prev_ndisplay
        if self.prev_ndisplay is not None:
            self.viewer.camera.center = self.prev_camera_center
        if self.prev_camera_zoom is not None:
            self.viewer.camera.zoom = self.prev_camera_zoom
        if self.prev_camera_angles is not None:
            self.viewer.camera.angles = self.prev_camera_angles
        self.slice_labels_layer = None
        self.slice_image_layer = None

    def manual_annotation(self):
        self.manual_annotation_dialog.show()

    def finish_manual_annotation(self):
        self.manual_annotation_dialog.hide()

    def threshold_annotation(self):
        threshold_slider = self.threshold_annotation_dialog.children()[2]
        threshold_slider.setMinimum(self.slice_image_layer.data.min())
        threshold_slider.setMaximum(self.slice_image_layer.data.max())
        threshold_slider.setValue((threshold_slider.minimum()+threshold_slider.maximum())/2)
        self.threshold_annotation_dialog.show()

    def threshold_value_changed(self, val):
        self.slice_labels_layer.data = self.slice_image_layer.data >= val

    def finish_threshold_annotation(self):
        self.threshold_annotation_dialog.hide()

    def minimal_contour_annotation(self):
        self.viewer.layers.selection.select_only(self.minimal_contour_widget.anchor_points)
        self.minimal_contour_widget.image_layer = self.slice_image_layer
        self.minimal_contour_widget.labels_layer = self.slice_labels_layer
        self.minimal_contour_widget.anchor_points.mode = "add"
        self.manual_annotation_dialog.show()

    def finish_minimal_contour_annotation(self):
        self.manual_annotation_dialog.hide()

    def initialize(self, image, phi_func, points, use_correction, beta, alpha, n_iter, use_meeting_plane_points, slice_annotation_method="Manual", blur_func=None, z_scale=1., use_gradient=True):
        self.image = image
        self.phi_func = phi_func
        self.points = points
        self.use_correction = use_correction
        self.beta = beta
        self.alpha = alpha
        self.n_iter = n_iter
        self.use_meeting_plane_points = use_meeting_plane_points
        self.blur_func = blur_func
        self.init_slice_annotation_fun = self.slice_annotation_methods[slice_annotation_method]
        self.finish_slice_annotation_fun = self.finish_annotation_methods[slice_annotation_method]
        self.z_scale = z_scale
        self.use_gradient = use_gradient

    def hook_callbacks(self, idx):
        estimator = self.estimators[idx]
        # estimator.hook_stage_data_init_event(
        #     minimal_surface.AREA_EIKONAL_STAGE,
        #     lambda arr, idx: self.data_initializer(
        #         "Area Eikonal %d%s" % (0, postscript),
        #         0,
        #         {
        #             'colormap': "plasma",
        #             # 'translate': offset,
        #             "visible": False
        #         }
        #     )(arr, idx)
        # )
        # estimator.hook_stage_iteration_event(minimal_surface.AREA_EIKONAL_STAGE,
        #                                      lambda idx: self.data_updater("Area Eikonal %d%s" % (0, postscript))(
        #                                          idx))
        # estimator.hook_stage_finished_event(minimal_surface.AREA_EIKONAL_STAGE,
        #                                     lambda: self.data_finalizer("Area Eikonal %d%s" % (0, postscript))())
        # estimator.hook_stage_data_init_event(
        #     minimal_surface.AREA_EIKONAL_STAGE,
        #     lambda arr, idx: self.data_initializer(
        #         "Area Eikonal %d%s" % (1, postscript),
        #         1,
        #         {
        #             'colormap': "plasma",
        #             # 'translate': offset,
        #             "visible": False
        #         }
        #     )(arr, idx)
        # )
        # estimator.hook_stage_iteration_event(minimal_surface.AREA_EIKONAL_STAGE,
        #                                      lambda idx: self.data_updater("Area Eikonal %d%s" % (1, postscript))(
        #                                          idx))
        # estimator.hook_stage_finished_event(minimal_surface.AREA_EIKONAL_STAGE,
        #                                     lambda: self.data_finalizer("Area Eikonal %d%s" % (1, postscript))())
        # self.hook_callbacks(estimator, minimal_surface.AREA_EIKONAL_STAGE, "Area Eikonal 0", {'colormap': "plasma", 'translate': offset, "visible": False}, 0)
        # self.hook_callbacks(estimator, minimal_surface.AREA_EIKONAL_STAGE, "Area Eikonal 1", {'colormap': "plasma", 'translate': offset, "visible": False}, 1)
        # self.hook_callbacks(estimator, minimal_surface.ROTATED_AREA_EIKONAL_STAGE, "Rotated Area Eikonal", {'colormap': "plasma"}, 0)
        # estimator.hook_stage_data_init_event(minimal_surface.PLANE_PHASEFIELD_STAGE,
        #                                      lambda arr, idx: print("plane phasefield eikonal init", arr.shape, idx))
        # estimator.hook_stage_data_init_event(
        #     minimal_surface.PLANE_PHASEFIELD_STAGE,
        #     lambda arr, idx: self.data_initializer("Plane PhaseField%s" % postscript,
        #                                            layer_args={
        #                                                'colormap': "plasma",
        #                                                'translate': offset + self.translation_matrix +
        #                                                             np.asarray([0, 0, self.center_mat[
        #                                                                 -1]]) @ self.rotation_matrix.T,
        #                                                'rotate': self.rotation_matrix,
        #                                                'opacity': 0.6,
        #                                                "visible": False
        #                                            }
        #                                            )(arr, idx)
        # )
        # estimator.hook_stage_iteration_event(minimal_surface.PLANE_PHASEFIELD_STAGE,
        #                                      lambda idx: self.data_updater("Plane PhaseField%s" % postscript)(idx))
        # estimator.hook_stage_finished_event(minimal_surface.PLANE_PHASEFIELD_STAGE,
        #                                     lambda: self.data_finalizer("Plane PhaseField%s" % postscript)())

        # estimator.hook_stage_data_init_event(minimal_surface.TRANSPORT_FUNCTION_STAGE,
        #                                      lambda arr, idx: print("plane phasefield eikonal init", arr.shape, idx))

        # estimator.hook_stage_data_init_event(
        #     minimal_surface.TRANSPORT_FUNCTION_STAGE,
        #     lambda arr, idx: self.data_initializer("Transport Function%s" % postscript,
        #                                            0,
        #                                            {
        #                                                'colormap': "turbo",
        #                                                'translate': offset + self.translation_matrix,
        #                                                'rotate': self.rotation_matrix,
        #                                                'opacity': 0.5,
        #                                                'rendering': "iso",
        #                                                "iso_threshold": 0.
        #                                            })(arr, idx)
        # )
        # # estimator.hook_stage_data_init_event(minimal_surface.TRANSPORT_FUNCTION_STAGE, lambda arr, idx: print("tf init"))
        # estimator.hook_stage_iteration_event(minimal_surface.TRANSPORT_FUNCTION_STAGE,
        #                                      lambda idx: self.data_updater("Transport Function%s" % postscript)(
        #                                          idx))
        # # estimator.hook_stage_iteration_event(minimal_surface.TRANSPORT_FUNCTION_STAGE, lambda idx: print("tf iter %d" % idx))
        # estimator.hook_stage_finished_event(minimal_surface.TRANSPORT_FUNCTION_STAGE,
        #                                     lambda: self.data_finalizer("Transport Function%s" % postscript)())

        def tform_calculated(rotation, translation):
            self.rotation_matrix = np.flip(rotation.copy()).reshape(3, 3)
            self.translation_matrix = np.flip(translation.copy())
        # print("hook_transform_calculated_event")
        # estimator.hook_transform_calculated_event(tform_calculated)
        # print("after hook_transform_calculated_event")

        def center_calculated(center):
            self.center_mat = np.flip(center.copy())
        # print("hook_plane_center_calculated_event")
        # estimator.hook_plane_center_calculated_event(center_calculated)
        # print("after hook_plane_center_calculated_event")

    def run(self):
        import time
        self.minimal_surface_widget.call_button.clicked.disconnect(self.minimal_surface_widget._start_estimation)
        def stop():
            self.stop_requested = True
            self.minimal_surface_widget.call_button.text = "Stopping..."
            self.finish_slice_annotation()
            self.minimal_surface_widget.call_button.enabled = True
        self.minimal_surface_widget.call_button.clicked.connect(stop)
        self.minimal_surface_widget.call_button.text = "Stop"
        self.minimal_surface_widget.call_button.enabled = False

        try:
            start = time.time()

            self.estimators = []
            self.bboxes = []
            self.imgs = []
            self.phis = []
            self.pt_pairs = []
            for i in range(len(self.points) // 2):
                estimator = minimal_surface.MinimalSurfaceCalculator()
                self.estimators.append(estimator)
                estimator.set_initial_plane_calculator(self.segment_initial_slice)
                estimator.set_using_meeting_points(self.use_meeting_plane_points)
                self.hook_callbacks(i)
                bounding_box = pts_2_bb(self.points[2 * i], self.points[2 * i + 1], self.image.shape,
                                        [self.z_scale, 1, 1])
                bounding_box = bounding_box.round().astype(int)
                self.bboxes.append(bounding_box)
                offset = bounding_box.min(0, keepdims=True)
                bb_slice = bb_2_slice(bounding_box)
                point1 = (self.points[2 * i] - offset).reshape(-1)
                point2 = (self.points[2 * i + 1] - offset).reshape(-1)
                point1[2] *= self.z_scale
                point2[2] *= self.z_scale
                self.pt_pairs.append((point1, point2))
                data = self.image[bb_slice]
                data = zoom(data, (self.z_scale, 1, 1))
                data = (data - (min_ := data.min())) / (data.max() - min_)
                if self.blur_func is not None:
                    data = self.blur_func(data)
                if self.use_gradient:
                    phi = sitk.GetArrayFromImage(sitk.GradientMagnitude(sitk.GetImageFromArray(data))).astype(float)
                else:
                    phi = self.phi_func(data)
                alpha = (self.alpha - np.exp(-self.beta))/(1-np.exp(-self.beta))
                phi = alpha + (1-alpha)*np.exp(-self.beta*phi)
                phi = phi/phi.max()
                self.imgs.append(data)
                self.phis.append(phi)
                if self.use_meeting_plane_points:
                    estimator.calc_eikonal_and_transport_init(phi, data, point1, point2, self.use_correction)
                else:
                    estimator.init_transport_slice(phi, point1, point2)
            self.slice_annotations_done.emit(len(self.estimators))

            show_info("Slice annotations finished in %.2f seconds" % (time.time() - start))
            timestamp = time.time()

            for i, (estimator, phi, data, (point1, point2), bounding_box)\
                    in enumerate(zip(self.estimators, self.phis, self.imgs, self.pt_pairs, self.bboxes)):
                if self.stop_requested:
                    show_info("Stopped calculation")
                    break
                if self.bb_layer is not None:
                    self.bb_layer.data = [bounding_box]

                offset = np.clip(bounding_box.min(0), 0, np.asarray(self.image.shape) - 1)
                # start = time.time()
                output = estimator.calculate(phi, data, point1, point2, self.use_correction, self.n_iter)
                segmented = (output >= 0)
                labelled = skimage.measure.label(segmented)
                obj_pixel = np.argwhere(output == output.max())[0]
                obj_label = labelled[tuple(obj_pixel)]
                mask = (labelled == obj_label)
                mask = zoom(mask, (1/self.z_scale, 1, 1), order=0)
                # self.image_data_received.emit("Result", output, {"colormap": "plasma", "translate": offset})
                self.mask_data_received.emit(mask, offset)
                postscript = " - %d" % (i + 1)
                # self.remove_layer.emit("Transport Function")
                self.object_annotated.emit(i)
            if not self.stop_requested:
                show_info("Estimation done in %d" % (time.time()-start))
        finally:
            self.stop_requested = False
            self.all_done.emit()
            self.minimal_surface_widget.call_button.clicked.disconnect(stop)
            self.minimal_surface_widget.call_button.clicked.connect(self.minimal_surface_widget._start_estimation)
            self.minimal_surface_widget.call_button.text = "Run"
            self.minimal_surface_widget.call_button.enabled = True

    def data_initializer(self, name, selected_idx=None, layer_args=None):
        if layer_args is None:
            layer_args = {}

        def initialize(arr, idx):
            try:
                if selected_idx in [None, idx]:
                    self.image_data_received.emit(name, arr, layer_args)
                else:
                    print("%s was not initialized as id (%d) was not %d" % (name, idx, selected_idx))
            except Exception as e:
                ...

        return initialize
        # return lambda arr, idx: print("initializing", name)

    def data_updater(self, name):
        it = [-1]

        def update_viewer(iteration):
            it[0] += 1
            if it[0] % 100 == 0: #TODO undo
                self.layer_invalidated.emit(name)
        return update_viewer

    def data_finalizer(self, name):
        def finalize():
            if name not in self.viewer.layers:
                print("%s not in layer list" % name)
                return
            self.viewer.layers[name].data = np.copy(self.viewer.layers[name].data)

        return finalize

    def segment_initial_slice(self, image_slice, distance_map):
        image_slice = np.squeeze(image_slice)
        distance_map = np.squeeze(distance_map)
        self.annotation_needed_signal.emit(image_slice, distance_map)
        while not self.done_pressed and not self.stop_requested:
            sleep(0.1)
            QApplication.processEvents()
        self.done_pressed = False
        if self.slice_annotation is None:
            raise ValueError("Slice not calculated")
        mask = self.slice_annotation.astype(float) if self.slice_annotation is not None else None
        self.slice_annotation = None
        return mask


class SliderWithCheckbox(FreeWidget):
    def __init__(self, state=False, value=None, min=None, max=None, enabled=True):
        super().__init__("horizontal")
        self.wdt = QWidget()
        layout = QVBoxLayout()
        self._checkbox = QCheckBox()
        layout.addWidget(self._checkbox)
        self._slider = QSymmetricDoubleRangeSlider(Qt.Horizontal)
        if min is not None:
            self.min = min
        if max is not None:
            self.max = max
        self.enabled = enabled
        if value is not None:
            self._slider.setValue(value)
        layout.addWidget(self._slider)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.wdt.setLayout(layout)
        self.set_widget(self.wdt)
        self.checked = state

    @property
    def checked(self):
        return self._checkbox.isChecked()

    @checked.setter
    def checked(self, new_value):
        if new_value == self._checkbox.isChecked():
            return
        self._checkbox.setChecked(new_value)
        self._slider.setEnabled(new_value)
        self.state_changed.emit(self.checked)

    @property
    def state_changed(self):
        return self._checkbox.clicked

    @property
    def value_changed(self):
        return self._slider.valueChanged

    @property
    def value(self):
        return self._slider.value()

    @value.setter
    def value(self, new_value):
        self._slider.setValue(new_value)

    @property
    def max(self):
        return self._slider.maximum()

    @max.setter
    def max(self, new_max):
        self._slider.setMaximum(new_max)

    @property
    def min(self):
        return self._slider.minimum()

    @min.setter
    def min(self, new_min):
        self._slider.setMinimum(new_min)


@magicclass(name="Minimal Surface", widget_type="scrollable", properties={"x_enabled": False})
class _MinimalSurfaceWidget(MagicTemplate):
    image_layer_combobox = field(Image, label="Image")
    labels_layer_combobox = field(Labels, label="Labels")

    @magicclass(widget_type="collapsible", name="Image Features")
    class ImageFeaturesWidget(MagicTemplate):
        ...

    @magicclass(widget_type="collapsible", name="Blurring")
    class BlurringWidget(MagicTemplate):
        def try_blurring(self): ...

    @magicclass(widget_type="collapsible", name="Slice Annotation Method")
    class SliceAnnotationWidget(MagicTemplate):
        ...

    @magicclass
    class ImageSliceWidget(MagicTemplate):
        dim_spinbox = field(int, label="Axis").with_options(max=0)
        position_slider = field(int, label="Position", widget_type="Slider").with_options(max=0)

        def __init__(self):
            super().__init__()
            self._viewer = None
            self._layer = None
            self._slice_layer = None
            self._ctrl_down = False
            self._mouse_drag_callbacks = []
            self._update_slice()
            self._update_widgets_state()
            self._update_max_position()
            self._clipping_widget = None

        @property
        def viewer(self):
            return self._viewer

        @viewer.setter
        def viewer(self, viewer):
            self._viewer = viewer
            viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)

        @property
        def clipping_widget(self):
            if self._clipping_widget is None:
                raise AttributeError("clipping_widget not initialized")
            return self._clipping_widget

        @clipping_widget.setter
        def clipping_widget(self, clipping_widget):
            if self._clipping_widget is not None:
                raise AttributeError("clipping_widget already initialized")
            self._clipping_widget = clipping_widget

        @property
        def position(self):
            return self.position_slider.value

        @property
        def dimension(self):
            return self.dim_spinbox.value

        @property
        def layer(self):
            return self._layer

        @layer.setter
        def layer(self, new_layer):
            self.clipping_widget.remove_layer(self._layer)
            self._layer = new_layer
            self.clipping_widget.add_layer(self._layer)
            self._update_slice()
            self._update_max_position()
            self._update_max_dim()
            self._update_widgets_state()
            if self._viewer.dims.ndisplay == 3:
                self._create_slice_layer()

        @dim_spinbox.connect
        def _on_dim_changed(self, new_dim):
            self._update_max_position()
            self._update_slice()
            self._update_widgets_state()

        def _update_max_position(self):
            if self.layer is None:
                return
            self.position_slider.max = self.layer.data.shape[self.dimension] - 1

        def _update_max_dim(self):
            if self.layer is None:
                return
            self.dim_spinbox.native.setMaximum(self.layer.data.ndim - 1)

        @position_slider.connect
        def _update_slice(self):
            if self._slice_layer is None:
                return
            dim = self.dimension
            idx = tuple(slice(self.position, self.position + 1) if i == dim else slice(None)
                        for i in range(self.layer.data.ndim))
            self._slice_layer.data = self.layer.data[idx]
            self._slice_layer.translate = [self.position * self.layer.scale[i] if i == dim else 0 for i in
                                           range(self.layer.data.ndim)]

        def _on_mouse_scroll(self, layer, event):
            if "Alt" not in event.modifiers:
                return
            self.position_slider.value = int(self.position - event.delta[0])

        def _update_widgets_state(self):
            is_enabled = self.layer is not None and self._viewer.dims.ndisplay == 3
            self.dim_spinbox.enabled = is_enabled
            self.position_slider.enabled = is_enabled

        def _remove_slice_layer(self):
            if self._slice_layer is not None:
                self._viewer.layers.remove(self._slice_layer)
                self.clipping_widget.remove_layer(self._slice_layer)
                self._slice_layer = None

        def _create_slice_layer(self):
            self._remove_slice_layer()
            if self.layer is None:
                return
            self._slice_layer = self._viewer.add_image(data=np.empty((1, 1, 1)), colormap="red",
                                                       scale=self.layer.scale, blending="additive",
                                                       experimental_clipping_planes=self.layer.experimental_clipping_planes)
            self.clipping_widget.add_layer(self._slice_layer)
            self._slice_layer.mouse_drag_callbacks.extend(self._mouse_drag_callbacks)
            self._slice_layer.mouse_wheel_callbacks.append(self._on_mouse_scroll)
            self._update_slice()
            self._update_widgets_state()
            self._slice_layer.reset_contrast_limits()

        def _on_ndisplay_changed(self):
            if self._viewer.dims.ndisplay == 2:
                self._remove_slice_layer()
            elif self._viewer.dims.ndisplay == 3:
                self._create_slice_layer()
            self._update_widgets_state()

        @set_design(visible=False)
        def add_mouse_drag_callback(self, callback):
            self._mouse_drag_callbacks.append(callback)
            if self._slice_layer is not None:
                self._slice_layer.mouse_drag_callbacks.append(callback)
            return callback

    call_button = field(widget_type=PushButton, name="Run")

    def _start_estimation(self):
        if self.image_layer not in self._viewer.layers:
            warnings.warn("Missing image layer")
            return
        if len(self.points_layer.data) == 0:
            warnings.warn("No points were added")
            return
        if len(self.points_layer.data) == 1:
            warnings.warn("Not enough points")
            return
        if self.labels_layer is None:
            answer = QMessageBox.question(
                self.native,
                "Missing Labels layer",
                "Create Labels layer?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if answer == QMessageBox.StandardButton.Yes:
                new_labels = self._viewer.add_labels(
                    np.zeros(self.image_layer.data.shape[:self.image_layer.ndim], dtype=np.uint16))
                self.labels_layer = new_labels
            else:
                return
        elif not np.array_equal(
                self.labels_layer.data.shape[:self.labels_layer.ndim],
                self.image_layer.data.shape[:self.image_layer.ndim]
        ):
            answer = QMessageBox.question(
                self.native,
                "Shape mismatch",
                "Current labels and image layers have different shape."
                " Do you want to create a new labels layer with the appropriate shape?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if answer == QMessageBox.StandardButton.Yes:
                new_labels = self._viewer.add_labels(
                    np.zeros(self.image_layer.data.shape[:self.image_layer.ndim], dtype=np.uint16))
                self.labels.layer = new_labels
            else:
                return
        image = self.image_layer
        image_data = image.data
        phi_func = None if (self.used_feature == "Gradient") else self._run_feature_extractor
        points = self.points_layer
        points_data = points.data.copy()
        z_scale = self.z_scale
        alpha = self.alpha
        beta = self.beta
        n_iter = self.iterations
        use_correction = self.use_correction
        use_gradient = self.used_feature == "Gradient"
        use_meeting_plane_points = True  # TODO check self.use_meeting_points_checkbox.isChecked()
        estimation_worker = EstimationWorker(self._viewer, self.minimal_contour_widget, self)
        estimation_worker.initialize(image_data, phi_func, points_data, use_correction, beta, alpha, n_iter, use_meeting_plane_points, self.slice_annotation_method, self._blur_image if self.is_blurring_enabled else None, z_scale, use_gradient)
        estimation_worker.image_data_received.connect(self._add_image)
        estimation_worker.layer_invalidated.connect(self._refresh_layer)
        estimation_worker.mask_data_received.connect(self._mask_received)
        estimation_worker.remove_layer.connect(self._data_remover)
        estimation_worker.slice_annotations_done.connect(self._show_estimation_progress)
        estimation_worker.object_annotated.connect(lambda i: self.estimation_progress_widget.setValue(i+1))
        estimation_thread = QThread(self.native)
        estimation_worker.moveToThread(estimation_thread)
        estimation_worker.all_done.connect(estimation_thread.quit)
        estimation_worker.all_done.connect(lambda: self.estimation_progress_widget.setVisible(False))
        estimation_thread.started.connect(estimation_worker.run)
        estimation_thread.finished.connect(estimation_worker.deleteLater)
        estimation_thread.finished.connect(estimation_thread.deleteLater)
        if BoundingBoxLayer is not None:
            estimation_worker.bb_layer = BoundingBoxLayer(
                [[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]], scale=(z_scale, 1., 1.))
            self._viewer.add_layer(estimation_worker.bb_layer)
            estimation_thread.finished.connect(lambda: self._viewer.layers.remove(estimation_worker.bb_layer))
        estimation_thread.start()
        # viewer.add_image(data)

    @magicclass(widget_type="collapsible")
    class ClippingWidget(MagicTemplate):
        slider_0 = field(SliderWithCheckbox)
        slider_1 = field(SliderWithCheckbox)
        slider_2 = field(SliderWithCheckbox)

        def __init__(self):
            super().__init__()
            self._viewer: napari.Viewer = None
            self._layers = []
            self._ghost_layers = dict()
            self.sliders = []
            self._clipping_planes = []
            self.n_checked = 0

        def __post_init__(self):
            for i, slider in enumerate([self.slider_0, self.slider_1, self.slider_2]):
                slider.name = str(i)
                slider.state_changed.connect(self._on_checked)
                slider.value_changed.connect(self._range_change_handler(i))
                self.sliders.append(slider)
            self._initialize_clipping_planes()

        @property
        def viewer(self):
            return self._viewer

        @viewer.setter
        def viewer(self, viewer):
            self._viewer = viewer
            viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)

        @property
        def layers(self):
            return list(self._layers)

        @layers.setter
        def layers(self, new_layers):
            new_layers = list(new_layers) if isinstance(new_layers, Iterable) \
                else [new_layers] if new_layers else []
            if self._layers:
                for layer in self.layers:
                    if layer not in new_layers:
                        self.remove_layer(layer)
                    else:
                        new_layers.remove(layer)
            for layer in new_layers:
                self.add_layer(layer)
            self._update_ranges()

        @set_design(visible=False)
        def add_layer(self, layer):
            if layer is None:
                return
            if layer in self.layers:
                return
            self._layers.append(layer)
            self._ghost_layers[layer] = None
            self._update_ghost_layers()
            layer.events.scale.connect(self._update_ranges)
            layer.experimental_clipping_planes = self._clipping_planes
            layer.events.data.connect(self._on_layer_data_change)
            self._update_ranges()

        @set_design(visible=False)
        def remove_layer(self, layer):
            if layer is None:
                return
            if layer not in self.layers:
                return
            self._layers.remove(layer)
            if self._ghost_layers[layer] is not None:
                self.viewer.layers.remove(self._ghost_layers[layer])
            del self._ghost_layers[layer]
            layer.events.scale.disconnect(self._update_ranges)
            layer.events.data.disconnect(self._on_layer_data_change)
            layer.experimental_clipping_planes = []
            self._update_ranges()

        def _on_layer_data_change(self, event):
            layer = event.source
            if layer not in self._ghost_layers:
                print("%s not in ghost layers!" % layer.name if layer else None)
                return
            ghost_layer = self._ghost_layers[event.source]
            if ghost_layer is None:
                return
            ghost_layer.data = event.source.data

        def _range_change_handler(self, idx):
            def on_range_change(value):
                self._clipping_planes[idx * 2]["position"][idx] = value[0]
                self._clipping_planes[idx * 2 + 1]["position"][idx] = value[1]
                self._update_clipping_planes()

            return on_range_change

        def _on_checked(self, value):
            self.n_checked += 1 if value else -1
            self._update_ghost_layers()
            self._update_clipping_planes()

        def _update_ghost_layers(self):
            if self.viewer is None:
                return
            for layer in self._layers:
                ghost = self._ghost_layers[layer]
                if ghost is None and self.n_checked > 0 and self.viewer.dims.ndisplay == 3:
                    data, state, type_str = layer.as_layer_data_tuple()
                    state["name"] = u"\U0001F47B " + layer.name
                    state["opacity"] /= 2.
                    state["experimental_clipping_planes"] = []
                    if NAPARI_VERSION <= version.parse("0.4.16") and "interpolation" in state:
                        interpolation = state["interpolation"]
                        state["interpolation"] = "nearest"
                    with layer_source(parent=layer):
                        new = Layer.create(deepcopy(data), state, type_str)
                    self.viewer.add_layer(new)
                    if NAPARI_VERSION <= version.parse("0.4.16") and "interpolation" in state:
                        new.interpolation = interpolation
                    self.viewer.layers.link_layers([layer, new])
                    self.viewer.layers.unlink_layers([layer, new],
                                                      ["opacity", "experimental_clipping_planes", "blending"])
                    self.viewer.layers.move(self.viewer.layers.index(new), self.viewer.layers.index(layer) + 1)
                    self._ghost_layers[layer] = new
                elif ghost is not None and (self.n_checked == 0 or self.viewer.dims.ndisplay == 2):
                    self.viewer.layers.remove(ghost)
                    self._ghost_layers[layer] = None

        def _remove_ghost_layers(self):
            for layer in self._layers:
                ghost = self._ghost_layers[layer]
                if ghost is not None:
                    self.viewer.layers.remove(ghost)
                    self._ghost_layers[layer] = None

        def _on_ndisplay_changed(self, e):
            if e.value == 2:
                self._remove_ghost_layers()
            else:
                self._update_ghost_layers()

        def _update_ranges(self):
            if not self.layers:
                return
            print("layers:", self.layers)
            nan = np.finfo(float).max
            negnan = np.finfo(float).min

            extent = np.asarray([[nan, nan, nan], [negnan, negnan, negnan]])
            for layer in self.layers:
                layer_extent = layer.extent.world
                if layer.ndim == 2:
                    layer_extent = np.concatenate([np.asarray([[nan], [negnan]]), layer_extent], axis=1)
                extent[0] = np.minimum(extent[0], layer_extent[0])
                extent[1] = np.maximum(extent[1], layer_extent[1])
            for s_idx, slider in enumerate(self.sliders):
                slider.checked = slider.checked or not np.any(np.equal(extent[:, s_idx], np.asarray([nan, negnan])))
                slider.min = extent[0, s_idx]
                slider.max = max(slider.min+1, extent[1, s_idx])
            self._update_clipping_planes()

        def _initialize_clipping_planes(self):
            new_clipping_planes = []
            for s_idx, slider in enumerate(self.sliders):
                start, end = slider.value
                clipping_plane0 = {
                    "position": [start if i == s_idx else 0 for i in range(3)],
                    "normal": [1 if i == s_idx else 0 for i in range(3)]
                }
                new_clipping_planes.append(clipping_plane0)
                clipping_plane1 = {
                    "position": [end if i == s_idx else 0 for i in range(3)],
                    "normal": [-1 if i == s_idx else 0 for i in range(3)]
                }
                new_clipping_planes.append(clipping_plane1)
            self._clipping_planes = new_clipping_planes
            for layer in self.layers:
                layer.experimental_clipping_planes = self._clipping_planes

        def _update_clipping_planes(self):
            new_clipping_planes = []
            for s_idx, slider in enumerate(self.sliders):
                if not slider.checked:
                    continue
                new_clipping_planes.extend(self._clipping_planes[s_idx * 2:s_idx * 2 + 2])
            for layer in self.layers:
                layer.experimental_clipping_planes = new_clipping_planes

    used_feature_combobox = field(widget_type="ComboBox", location=ImageFeaturesWidget).with_choices(FEATURE_KEYS)
    feature_editor = field(ScriptExecuteWidget, location=ImageFeaturesWidget).with_options(editor_key="minimal_surface_features")
    alpha = vfield(float, label="\u03B1", widget_type="FloatSlider", location=ImageFeaturesWidget).with_options(min=0, max=1, value=0.01)
    beta = vfield(float, label="\u03B2", location=ImageFeaturesWidget).with_options(min=.01, max=500., value=5)
    iterations = vfield(int, widget_type="Slider", location=ImageFeaturesWidget).with_options(min=1, max=10000, value=10000)
    z_scale = vfield(float, location=ImageFeaturesWidget).with_options(min=0.01, max=20., value=1.)
    use_correction = vfield(bool, location=ImageFeaturesWidget)

    blur_image_checkbox = field(bool, label="Blur image", location=BlurringWidget)
    blurring_type_combobox = (field(str, widget_type="ComboBox", location=BlurringWidget)
                              .with_choices([GAUSSIAN_BLURRING, CAD_BLURRING]))
    blur_sigma_slider = field(float, widget_type="FloatSlider", location=BlurringWidget).with_options(max=20)
    conductance_spinbox = field(float, location=BlurringWidget).with_options(max=100, value=9)
    cad_iterations_spinbox = field(int, label="Iterations", location=BlurringWidget).with_options(min=1, max=20, value=5)

    slice_annotation_method = vfield(str, location=SliceAnnotationWidget).with_choices(EstimationWorker.ANNOTATION_METHODS)

    def __init__(self):
        self._collapsible_group = CollapsibleContainerGroup()
        self._viewer = None
        self._prev_labels = None
        self.blurred_layer = None
        self._try_blurring_button = None
        self.blurring_progress_widget = ProgressWidget(message="blurring...")
        self.estimation_progress_widget = ProgressWidget(message="Annotating...")
        self.points_layer = None
        self._estimation_running = False
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        # Parameters for the estimator

        # self.z_scale_spinbox.valueChanged.connect(self) TODO

        # Blurring parameters
        # delayed_updater.processing.connect(
        #     lambda _: self.blurring_progress_widget.setVisible(blur_demo_btn.isChecked())) # TODO
        # delayed_updater.processed.connect(lambda _: self.blurring_progress_widget.setVisible(False)) # TODO

        self.crop_mask = None

        # self.use_meeting_points_checkbox = QCheckBox("Slice from meeting points") #TODO check if needed
        # self.use_meeting_points_checkbox.setChecked(True)
        # self.slice_widget.clipping_widget.add_layer(self.points_layer)

    def __post_init__(self):
        self.call_button.clicked.connect(self._start_estimation)
        print(self.native.children()[1].sizeHint().width())
        print(self.native.children()[1].verticalScrollBar().sizeHint().width())
        print(self.native.children()[1].widget().layout().getContentsMargins()[2])
        self.native.setMinimumWidth(self.native.children()[1].widget().sizeHint().width()
                                    + self.native.children()[1].verticalScrollBar().sizeHint().width()
                                    + 2 * self.native.children()[1].widget().layout().getContentsMargins()[2])
        for container in [self.ImageFeaturesWidget, self.BlurringWidget, self.SliceAnnotationWidget]:
            container._widget._expand_btn.setChecked(False)
            # container._widget._inner_qwidget.setStyleSheet("padding: 0 10 0 10")
            container.collapsed = True
            self._collapsible_group.addItem(container)
        self._on_feature_changed(self.used_feature)
        self.try_blurring_button.native.setCheckable(True)
        self._on_blur_checked(self.blur_image_checkbox.value)
        self._on_labels_changed()
        self._ensure_3d_image()

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
    def labels_layer(self):
        return self.labels_layer_combobox.value

    @labels_layer.setter
    def labels_layer(self, new_layer):
        self.labels_layer_combobox.value = new_layer

    @property
    def is_blurring_enabled(self):
        return self.blur_image_checkbox.value

    @property
    def used_feature(self):
        return self.used_feature_combobox.value

    @property
    def blurring_type(self):
        return self.blurring_type_combobox.value

    @property
    def blurring_conductance(self):
        return self.conductance_spinbox.value

    @property
    def blurring_iterations(self):
        return self.cad_iterations_spinbox.value

    @property
    def try_blurring_button(self):
        if self._try_blurring_button is None:
            for w in self._list:
                if w.name == "try_blurring":
                    self._try_blurring_button = w
                    break
        return self._try_blurring_button

    def _initialize(self, viewer: napari.Viewer, minimal_contour_widget: MinimalContourWidget):
        self._viewer = viewer
        self.minimal_contour_widget = minimal_contour_widget
        self.points_layer = self.viewer.add_points(ndim=3, size=2, name="3D Anchors [DO NOT REMOVE]")
        self.points_layer.events.connect(ColorPairsCallback())
        self.points_layer.mouse_drag_callbacks.append(self._drag_points_callback)
        self.viewer.layers.events.inserted.connect(keep_layer_on_top(self.points_layer))
        self.viewer.layers.events.moved.connect(keep_layer_on_top(self.points_layer))
        self.ImageSliceWidget.viewer = self.viewer
        self.ImageSliceWidget.clipping_widget = self.ClippingWidget
        self._ensure_3d_image()
        self._set_slice_widget_image()
        self._update_custom_feature_image()
        self.ImageSliceWidget.add_mouse_drag_callback(self._on_slice_clicked)
        self.ClippingWidget.viewer = self.viewer

    @set_design(text="Try", location=BlurringWidget,)
    def try_blurring(self):
        self._toggle_blur_demo()

    @labels_layer_combobox.connect
    def _on_labels_changed(self, idx=None):
        if self._prev_labels == self.labels_layer:
            return
        self.ImageSliceWidget.clipping_widget.remove_layer(self._prev_labels)
        self.ImageSliceWidget.clipping_widget.add_layer(self.labels_layer)
        self._prev_labels = self.labels_layer

    @used_feature_combobox.connect
    def _on_feature_changed(self, feature):
        self.feature_editor.visible = feature == "Custom"
        correct_container_size(self.ImageFeaturesWidget)

    def _add_image(self, layer_name, data, layer_args):
        if 'opacity' not in layer_args:
            layer_args['opacity'] = 0.3
        if 'blending' not in layer_args:
            layer_args['blending'] = "additive"
        if 'rgb' not in layer_args:
            layer_args['rgb'] = False
        self._viewer.add_layer(Image(data, name=layer_name, **layer_args))

    def _mask_received(self, data, offset):
        offset = offset.round().astype(int)
        bb_slice = tuple(slice(offs, offs + extent) for offs, extent in zip(offset, data.shape))
        self.labels_layer.data[bb_slice][data > 0] = self.labels_layer.data.max()+1
        self.labels_layer.refresh()

    def _refresh_layer(self, layer_name):
        if layer_name not in self._viewer.layers:
            return
        layer = self._viewer.layers[layer_name]
        # layer.data = layer.data
        layer.refresh()

    def _data_remover(self, name):
        self._viewer.layers.remove(name)

    def _show_estimation_progress(self, n_objects):
        self.estimation_progress_widget.setMaximum(n_objects)
        self.estimation_progress_widget.setValue(0)
        self.estimation_progress_widget.setVisible(True)

    def _run_feature_extractor(self, image):
        self.feature_editor.variables["image"] = image
        res = dict()

        def store_features(result_dict):
            res.update(result_dict)
        self.feature_editor.script_worker.done.connect(store_features, Qt.DirectConnection)
        self.feature_editor.Run()
        self.feature_editor.script_thread.wait()
        if "exception" in res:
            raise res["exception"]
        return res["features"]

    # def hook_callbacks(self, estimator, stage, layer_name, layer_params, data_idx=None):
    #
    #     def initializer(arr, idx):
    #         self.data_initializer(layer_name, data_idx, layer_params)(arr, idx)
    #
    #     estimator.hook_stage_data_init_event(stage, initializer)
    #     estimator.hook_stage_iteration_event(stage, self.data_updater(layer_name))
    #     estimator.hook_stage_finished_event(stage, self.data_finalizer(layer_name))

    def _drag_points_callback(self, layer, event):
        if layer.mode != "select":
            return
        yield
        if event.type == "mouse_move":
            if len(layer.selected_data) > 2 or len(layer.selected_data) == 0:
                return
            if len(layer.selected_data) == 2:
                selection_iter = iter(layer.selected_data)
                idx1 = next(selection_iter)
                idx2 = next(selection_iter)
                if idx1 // 2 != idx2 // 2:
                    return
            else:
                index = next(iter(layer.selected_data))
                if index % 2 == 0:
                    idx1 = index
                    idx2 = index + 1
                else:
                    idx1 = index - 1
                    idx2 = index
            p1 = layer.data[idx1]
            p2 = layer.data[idx2]
            bb = pts_2_bb(p1, p2, self.image_layer.data.shape)
            bb_slice = bb_2_slice(bb)
            image = self.image_layer.data[bb_slice]
            crop_mask = generate_sphere_mask(image.shape)
            self.image_layer.visible = False
            cropped_image = self._viewer.add_image(
                image*crop_mask,
                colormap=self.image_layer.colormap,
                translate=bb.min(0)*self.image_layer.scale,
                scale=self.image_layer.scale,
                rendering="attenuated_mip"
            )
            self._viewer.layers.selection.active = layer
        yield
        while event.type == "mouse_move":
            yield
        self._viewer.layers.remove(cropped_image)
        self._update_blurred_layer()
        self.image_layer.visible = True

    @blurring_type_combobox.connect
    def _on_blur_selection(self, blur_method):
        self.blur_sigma_slider.visible = blur_method == GAUSSIAN_BLURRING
        self.conductance_spinbox.visible = blur_method == CAD_BLURRING
        self.cad_iterations_spinbox.visible = blur_method == CAD_BLURRING
        correct_container_size(self.BlurringWidget)

    @blur_image_checkbox.connect
    def _on_blur_checked(self, is_checked):
        self.blurring_type_combobox.visible = is_checked
        correct_container_size(self.BlurringWidget)
        self._on_blur_selection(self.blurring_type_combobox.value if is_checked else "")

    def _blur_image(self, image):
        blur_type = self.blurring_type
        if blur_type == GAUSSIAN_BLURRING:
            return gaussian_filter(image, self.blur_sigma_slider.value)
        elif blur_type == CAD_BLURRING:
            sitk_image = sitk.GetImageFromArray(image.astype(float))
            blurred = sitk.CurvatureAnisotropicDiffusion(
                sitk_image,
                conductanceParameter=self.blurring_conductance,
                numberOfIterations=self.blurring_iterations
            )
            return sitk.GetArrayFromImage(blurred)
        elif blur_type != "":
            raise NotImplementedError("Unkown blur type %s" % blur_type)

    def _toggle_blur_demo(self):
        is_checked = not self.try_blurring_button.native.isChecked()
        self.try_blurring_button.native.setChecked(is_checked)
        print(is_checked)
        if len(self.points_layer.data) < 2:
            return
        image_layer = self.image_layer
        if image_layer is None:
            return
        if is_checked:
            bb = pts_2_bb(self.points_layer.data[0], self.points_layer.data[1], image_layer.data.shape, image_layer.scale)
            bb = bb.round().astype(int)
            bb_slice = bb_2_slice(bb)
            blurred_image = image_layer.data[bb_slice]
            self.crop_mask = generate_sphere_mask(blurred_image.shape)
            offset = bb.min(0)

            blurred = self._blur_image(blurred_image)
            blurred *= self.crop_mask
            self.blurred_layer = self._viewer.add_image(
                blurred,
                name="%s cropped" % image_layer.name,
                translate=np.add(image_layer.translate, offset),
                scale=image_layer.scale,
                colormap=image_layer.colormap,
                contrast_limits=image_layer.contrast_limits
            )
        else:
            if self.blurred_layer:
                self._viewer.layers.remove(self.blurred_layer)
                self.blurred_layer = None
            self.crop_mask = None

    @blurring_type_combobox.connect
    @conductance_spinbox.connect
    @cad_iterations_spinbox.connect
    @blur_sigma_slider.connect
    @delay_function
    def _update_blurred_layer(self, *args):
        if self.blurred_layer is None:
            return
        image_layer = self.image_layer
        if image_layer is None:
            return
        bb = pts_2_bb(self.points_layer.data[0], self.points_layer.data[1], image_layer.data.shape, image_layer.scale)
        bb = bb.round().astype(int)
        offset = bb.min(0)
        bb_slice = bb_2_slice(bb)
        blurred_image = image_layer.data[bb_slice]
        if self.crop_mask is None or self.crop_mask.shape != blurred_image.shape:
            self.crop_mask = generate_sphere_mask(blurred_image.shape)
        blurred = self._blur_image(blurred_image)
        blurred *= self.crop_mask
        self.blurred_layer.data = blurred
        self.blurred_layer.translate = np.add(image_layer.translate, offset)

    @image_layer_combobox.connect
    def _ensure_3d_image(self, _=None):
        if self.image_layer is None:
            return
        if self.image_layer.ndim == 2:
            warnings.warn("Minimal Surface works only with 3D images!")
        is_enabled = self.image_layer is not None and self.image_layer.ndim==3
        print("is enabled: ", is_enabled)
        for container in self._collapsible_group:
            container.enabled = is_enabled
        self.ClippingWidget.enabled = is_enabled
        self.call_button.enabled = is_enabled

    @image_layer_combobox.connect
    def _set_slice_widget_image(self, _=None):
        self.ImageSliceWidget.layer = self.image_layer if self.image_layer is None or self.image_layer.ndim == 3 else None

    @image_layer_combobox.connect
    def _update_custom_feature_image(self, _=None):
        if self.image_layer is not None and self.image_layer.ndim != 2:
            self.feature_editor.variables["image"] = self.image_layer.data

    def _on_slice_clicked(self, layer, event):
        start_point, end_point = layer.get_ray_intersections(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        dragging = False
        point = None
        if (start_point is not None) and (end_point is not None):
            point = np.add(start_point, end_point)/2
            point = layer.data_to_world(point)
        yield
        while event.type == "mouse_move":
            dragging = True
            yield
        if dragging:
            return
        if event.button == 2:
            self.points_layer.remove_selected()
            return
        if point is not None:
            if QApplication.keyboardModifiers() & Qt.AltModifier:
                for i, slider in enumerate(self.ImageSliceWidget.clipping_widget.sliders):
                    range = slider.value
                    range_center = (range[0] + range[1])/2
                    new_center = int(point[i])
                    offset = new_center-range_center
                    slider.value = (range[0]+offset, range[1]+offset)
            else:
                self.points_layer.add(self.points_layer.world_to_data(point))


class ShortestPathsWidget(WidgetWithLayerList):
    shapes_data_received = Signal(str, "PyQt_PyObject", "PyQt_PyObject")

    def __init__(self, viewer: napari.Viewer):
        super().__init__(viewer, [("meeting_plane", Image), ("distance_map", Image), ("starting_points", Points)], "nd_annotator_shortest_paths")
        self.shapes_data_received.connect(self.add_shapes)
        self.viewer = viewer
        self.estimator = minimal_surface.MinimalSurfaceCalculator()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Extract contour from meeting plane"))
        plane_2_contour_btn = QPushButton("Extract")
        plane_2_contour_btn.clicked.connect(self.plane_2_contour)
        layout.addWidget(plane_2_contour_btn)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #414851;")
        layout.addWidget(line)

        run_button = QPushButton("Show shortest paths")
        run_button.clicked.connect(self.show_shortest_paths)
        layout.addWidget(run_button)
        self.setLayout(layout)

    def add_shapes(self, layer_name, data, layer_args):
        if 'opacity' not in layer_args:
            layer_args['opacity'] = 0.3
        self.viewer.add_layer(Shapes(data, name=layer_name, shape_type="path", **layer_args))

    def plane_2_contour(self):
        if self.meeting_plane.layer is None:
            return
        meeting_plane_layer = self.meeting_plane.layer
        plane_mask = (meeting_plane_layer.data > 0).astype(np.uint8)
        contour = np.squeeze(cv2.findContours(np.squeeze(plane_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0])
        contour = np.concatenate([np.fliplr(contour), np.ones([len(contour), 1])], axis=1)
        contour = contour @ meeting_plane_layer.rotate.T + meeting_plane_layer.translate
        contour_layer = Points(contour, ndim=3, name="contour points", size=2)
        self.viewer.add_layer(contour_layer)
        self.starting_points.layer = contour_layer

    def calculate(self, points, distance_map, translate, rotate):
        paths = []
        for point in points:
            shortest_path = self.estimator.resolve_shortest_paths(point, distance_map).copy()
            paths.append(shortest_path)
        colors = generate_label_colors(len(paths))
        self.shapes_data_received.emit("Shortest path", paths, {
            'translate': translate,
            'rotate': rotate,
            'edge_color': np.concatenate([np.asarray(colors) / 255, np.ones([len(colors), 1])], axis=1),
            'edge_width': 0.1
        })

    def show_shortest_paths(self):
        distance_map = self.distance_map.layer
        points = self.starting_points.layer
        if distance_map is None or points is None:
            return
        translation = distance_map.translate
        rotation = distance_map.rotate
        distance_map = distance_map.data
        points = (points.data - translation).astype(int)
        self.viewer.window.worker2 = Thread(target=self.calculate, args=[points, distance_map, translation, rotation])
        self.viewer.window.worker2.start()

if minimal_surface is not None:
    import SimpleITK as sitk
    MinimalSurfaceWidget = _MinimalSurfaceWidget
else:
    MinimalSurfaceWidget = None

