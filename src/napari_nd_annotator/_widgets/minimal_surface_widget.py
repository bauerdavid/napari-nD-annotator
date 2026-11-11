import warnings

from magicgui.widgets import PushButton
from time import sleep
import numpy as np
import itertools
import colorsys
from random import shuffle, seed

#image processing modules
import skimage.measure, skimage.morphology
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, zoom

#UI
from magicclass import magicclass, MagicTemplate, field, vfield, set_design
from magicclass.serialize import serialize

from qtpy.QtCore import Signal, Qt, QObject, QThread
from qtpy.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QApplication
)

import napari
from napari.layers import Image, Points, Shapes
from napari.layers.labels._labels_constants import Mode
from napari.utils.notifications import show_info


from ._utils import ProgressWidget, QDoubleSlider, ScriptExecuteWidget
from ._utils.callbacks import keep_layer_on_top
from ._utils.clipping import SCAFFOLD_KEY
from .._helper_functions import layer_dims_displayed, viewer_canvas_size, viewer_camera, unwrap_viewer
from ..surface_anchors import suggest_anchors_in_image
from ..object_anchors import anchor_store, pick_along_ray, pick_on_plane
from .object_list import selected_object_changed, selected_object_info
from ._utils.collapsible_widget import CollapsibleContainerGroup, correct_container_size
from .minimal_contour_widget import MinimalContourWidget, delay_function


try:
    from napari_bbox import BoundingBoxLayer
except ImportError:
    BoundingBoxLayer = None

try:
    import minimal_surface
except ImportError:
    minimal_surface = None


GRADIENT = "Gradient"
GRADIENT_INTENSITY = "Gradient + intensity"
CUSTOM = "Custom"
FEATURE_KEYS = [GRADIENT, GRADIENT_INTENSITY, CUSTOM]
# FEATURES = {
#     ASSYM_GRADIENT: GRADIENT_BASED,
#     SYM_GRADIENT: INTENSITY_BASED,
#     HIGH_INTENSITY: INTENSITY_BASED,
#     LOW_INTENSITY: INTENSITY_BASED,
#     CUSTOM: INTENSITY_BASED
# }

# the estimator's correction is always on
USE_CORRECTION = True

IMAGE_PLANE = "Image plane"
ANCHOR_PLACEMENTS = ["Ray pick", IMAGE_PLANE]
ANCHOR_COLORS = ["red", "cyan"]

SUGGESTION_MODES = {"Axis": "axis", "Fan (farthest pair)": "fan"}

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


def write_object_mask(labels_layer, label, mask, offset, bbox=None):
    """Writes a segmentation result into a labels layer as the object ``label``, as one undoable edit.

    ``mask`` is a boolean array whose corner is at the index ``offset``. The voxels under the mask take the label
    whatever they held (as when painting). The previous mask of the object is replaced: the voxels of ``label`` inside
    the region this touches, the union of the box of ``mask`` and ``bbox`` (the inclusive corners of the object's
    bounding box), become background. Voxels of the object further away stay.
    """
    offset = np.asarray(offset).round().astype(int)
    shape = np.asarray(labels_layer.data.shape)
    low = np.clip(offset, 0, shape)
    high = np.clip(offset + np.asarray(mask.shape), 0, shape)
    if bbox is not None:
        low = np.minimum(low, np.clip(np.floor(bbox[0]).astype(int), 0, shape))
        high = np.maximum(high, np.clip(np.floor(bbox[1]).astype(int) + 1, 0, shape))
    region_slice = tuple(slice(a, b) for a, b in zip(low, high))
    region = np.asarray(labels_layer.data[region_slice])
    updated = region.copy()
    updated[region == label] = 0
    mask_slice = tuple(slice(max(o, a) - o, min(o + m, b) - o) for o, m, a, b in zip(offset, mask.shape, low, high))
    target_slice = tuple(slice(max(o, a) - a, min(o + m, b) - a) for o, m, a, b in zip(offset, mask.shape, low, high))
    updated[target_slice][np.asarray(mask)[mask_slice] > 0] = label
    changed = np.nonzero(updated != region)
    if len(changed[0]) == 0:
        return
    indices = tuple(index + start for index, start in zip(changed, low))
    labels_layer.data_setitem(indices, updated[changed])


def gradient_intensity_features(image, delta=.1):
    """High where the gradient is high and the intensity is low, in 0..1 (the feature of the minimal-surface README).

    Unlike the plain gradient it tells the object from the background, which the automatic slice annotation needs.
    """
    low, high = image.min(), np.quantile(image, .95)
    image = np.clip((image - low) / (high - low), 0, 1) if high > low else np.zeros_like(image, dtype=float)
    gradient = gaussian_gradient_magnitude(image, 1.)
    features = gradient * (1 + (delta - 1) * image ** 2) + delta * (1 - image ** 2)
    span = features.max() - features.min()
    return (features - features.min()) / span if span > 0 else np.zeros_like(features)


def annotation_methods():
    """The slice annotation methods; the automatic ones need a minimal-surface build that has them."""
    methods = list(EstimationWorker.ANNOTATION_METHODS)
    if minimal_surface is not None and hasattr(minimal_surface.MinimalSurfaceCalculator, "set_initial_plane_method"):
        methods += list(EstimationWorker.AUTOMATIC_METHODS)
    return methods


class EstimationWorker(QObject):
    image_data_received = Signal(str, object, object)
    mask_data_received = Signal(object, object)
    distance_map_received = Signal(str, int, object, object)
    contour_points_received = Signal(str, object, object)
    remove_layer = Signal(str)
    layer_invalidated = Signal(str)
    all_done = Signal()
    annotation_needed_signal = Signal(object, object)
    slice_annotations_done = Signal(int)
    object_annotated = Signal(int)
    MANUAL_ANNOTATION = "Manual"
    THRESHOLD_ANNOTATION = "Threshold"
    MINIMAL_CONTOUR_ANNOTATION = "Minimal contour"
    DISTANCE_AUTOMATIC = "Automatic (distances)"
    PHASE_FIELD_AUTOMATIC = "Automatic (phase field)"
    # the methods that find the contour in the meeting plane without the user, by the name the estimator knows them by
    AUTOMATIC_METHODS = {DISTANCE_AUTOMATIC: "distance", PHASE_FIELD_AUTOMATIC: "phase_field"}
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
        self.beta = None
        self.alpha = None
        self.n_iter = None
        self.use_meeting_plane_points = None
        self.done_pressed = False
        self.blur_func = None
        self.init_slice_annotation_fun = None
        self.finish_slice_annotation_fun = None
        self.initial_plane_method = None
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
        self.prev_camera_center = viewer_camera(self.viewer).center
        self.prev_camera_zoom = viewer_camera(self.viewer).zoom
        self.prev_camera_angles = viewer_camera(self.viewer).angles
        self.viewer.dims.ndisplay = 2
        visible_extent = self.slice_labels_layer.extent.world[:, layer_dims_displayed(self.slice_labels_layer)]
        viewer_camera(self.viewer).center = visible_extent.mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            viewer_camera(self.viewer).zoom = np.min(np.divide(viewer_canvas_size(self.viewer), visible_extent.max(0)-visible_extent.min(0)))*0.95
        self.slice_labels_layer.brush_size = 1
        self.slice_labels_layer.mode = "paint"
        self.minimal_surface_widget.call_button.enabled = True
        self.init_slice_annotation_fun()

    def finish_slice_annotation(self):
        if self.finish_slice_annotation_fun is not None:
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
            viewer_camera(self.viewer).center = self.prev_camera_center
        if self.prev_camera_zoom is not None:
            viewer_camera(self.viewer).zoom = self.prev_camera_zoom
        if self.prev_camera_angles is not None:
            viewer_camera(self.viewer).angles = self.prev_camera_angles
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
        # the widget follows the active labels layer; its source image (the slice image) was recorded when it was added
        self.viewer.layers.selection.select_only(self.slice_labels_layer)
        # the minimal contour tool borrows the pan/zoom mode, like the button and the "0" shortcut do
        self.slice_labels_layer.mode = Mode.PAN_ZOOM
        self.minimal_contour_widget._enable_mc_mode()
        self.manual_annotation_dialog.show()

    def finish_minimal_contour_annotation(self):
        self.manual_annotation_dialog.hide()

    def initialize(self, image, phi_func, points, beta, alpha, n_iter, use_meeting_plane_points, slice_annotation_method="Manual", blur_func=None, z_scale=1., use_gradient=True):
        self.image = image
        self.phi_func = phi_func
        self.points = points
        self.beta = beta
        self.alpha = alpha
        self.n_iter = n_iter
        self.use_meeting_plane_points = use_meeting_plane_points
        self.blur_func = blur_func
        self.initial_plane_method = EstimationWorker.AUTOMATIC_METHODS.get(slice_annotation_method)
        self.init_slice_annotation_fun = self.slice_annotation_methods.get(slice_annotation_method)
        self.finish_slice_annotation_fun = self.finish_annotation_methods.get(slice_annotation_method)
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

        anchor_maps = {}

        def anchor_map_initialized(distance_map, anchor_idx):
            anchor_maps[anchor_idx] = distance_map

        def anchor_maps_finished():
            # the arrays are views of the estimator's memory, which is reused afterwards
            for anchor_idx, distance_map in anchor_maps.items():
                self.distance_map_received.emit(
                    "Distance map - %d (anchor %d)" % (idx + 1, anchor_idx + 1),
                    anchor_idx,
                    distance_map.copy(),
                    {"translate": self.bboxes[idx].min(0), "scale": (1 / self.z_scale, 1., 1.)})

        estimator.hook_stage_data_init_event(minimal_surface.AREA_EIKONAL_STAGE, anchor_map_initialized)
        estimator.hook_stage_finished_event(minimal_surface.AREA_EIKONAL_STAGE, anchor_maps_finished)

        def tform_calculated(rotation, translation):
            self.rotation_matrix = np.flip(rotation.copy()).reshape(3, 3)
            self.translation_matrix = np.flip(translation.copy())
        estimator.hook_transform_calculated_event(tform_calculated)

        def center_calculated(center):
            self.center_mat = np.flip(center.copy())
        estimator.hook_plane_center_calculated_event(center_calculated)

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
                if self.initial_plane_method is None:
                    estimator.set_initial_plane_calculator(self.segment_initial_slice)
                else:
                    estimator.set_initial_plane_method(self.initial_plane_method)
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
                point1[0] *= self.z_scale
                point2[0] *= self.z_scale
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
                    estimator.calc_eikonal_and_transport_init(phi, data, point1, point2, USE_CORRECTION)
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
                output = estimator.calculate(phi, data, point1, point2, USE_CORRECTION, self.n_iter)
                segmented = (output >= 0)
                labelled = skimage.measure.label(segmented)
                obj_pixel = np.argwhere(output == output.max())[0]
                obj_label = labelled[tuple(obj_pixel)]
                mask = (labelled == obj_label)
                orig_z = bounding_box[:, 0].max() - bounding_box[:, 0].min()
                mask = zoom(mask, (orig_z / mask.shape[0], 1, 1), order=0)
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
        self.contour_points_received.emit(
            "Contour points - %d" % len(self.bboxes),
            self.plane_contour_points(mask),
            {"translate": self.bboxes[-1].min(0), "scale": (1 / self.z_scale, 1., 1.)})
        return mask

    def plane_contour_points(self, mask):
        # the contour of the annotated plane, in the bounding box frame of the distance map
        contours = skimage.measure.find_contours(mask, 0.5)
        if not contours:
            return None
        points = np.concatenate(contours)
        # the plane is (rows, columns, 1): its last axis is the plane normal, at the plane center
        points = np.concatenate([points, np.full([len(points), 1], self.center_mat[-1])], axis=1)
        return points @ self.rotation_matrix.T + self.translation_matrix


@magicclass(name="Minimal Surface", widget_type="scrollable", properties={"x_enabled": False})
class _MinimalSurfaceWidget(MagicTemplate):
    image_layer_combobox = field(Image, label="Image")
    object_status_label = field(widget_type="Label", label="Object")
    place_anchors_checkbox = field(bool, label="Place anchors in 3D").with_options(
        value=True,
        tooltip="In the 3D view, click (without dragging) inside the bounding box of the selected object to place its"
                " anchors on the surface, right-click to remove the last one. A third click starts a new pair."
                " Only while this tab is shown and the object list has an object selected."
    )
    anchor_placement_combobox = field(widget_type="ComboBox", label="Placement").with_options(
        choices=ANCHOR_PLACEMENTS,
        tooltip="Ray pick: the anchor is where the ray under the cursor first enters the image data inside the"
                " bounding box. Image plane: the anchor is where the ray crosses the plane of the image layer"
                " (shown with the plane depiction: Shift+drag moves it, double click sets its position)."
    )
    clear_anchors_button = field(widget_type=PushButton, name="Clear anchors").with_options(
        tooltip="Remove the anchors of the selected object"
    )

    @magicclass(widget_type="collapsible", name="Image Features")
    class ImageFeaturesWidget(MagicTemplate):
        ...

    @magicclass(widget_type="collapsible", name="Blurring")
    class BlurringWidget(MagicTemplate):
        def try_blurring(self): ...

    @magicclass(widget_type="collapsible", name="Slice Annotation Method")
    class SliceAnnotationWidget(MagicTemplate):
        ...

    suggestion_mode_combobox = field(widget_type="ComboBox", label="Suggestion mode").with_options(
        choices=list(SUGGESTION_MODES),
        tooltip="Axis: the anchors are on the boundary along the longest axis of the bounding box."
                " Fan: casts rays in all directions and takes the farthest pair of boundary points,"
                " also for objects that are tilted in the box (can be less robust next to touching objects)."
    )
    fan_rays_spinbox = field(int, label="Fan: rays").with_options(
        min=16, max=2000, step=16, value=256,
        tooltip="Number of directions the fan casts rays in from the center of the bounding box"
    )
    fan_max_radius_jump_spinbox = field(float, label="Fan: max radius jump").with_options(
        min=1., max=5., step=.1, value=1.3,
        tooltip="A ray whose boundary is farther from the center than this factor times that of its neighboring"
                " rays is ignored (leak into a neighbor or a noisy far edge). Lower: stricter, but too low"
                " cuts off the tips of elongated objects"
    )
    fan_min_relative_strength_spinbox = field(float, label="Fan: min relative strength").with_options(
        min=0., max=2., step=.1, value=.8,
        tooltip="Rays with a boundary weaker than this fraction of the typical one are ignored."
                " Higher: only the clearest boundaries count"
    )
    suggest_anchors_button = field(widget_type=PushButton, name="Suggest anchors").with_options(
        tooltip="Propose the two anchor points from the bounding box selected in the object list."
                " Click in the 3D view to place them again, then run."
    )
    call_button = field(widget_type=PushButton, name="Run")

    def _suggest_anchors(self):
        if self.call_button.text != "Run":
            warnings.warn("Wait until the estimation finishes before suggesting anchors")
            return
        selected = self._selected_object()
        if selected is None:
            warnings.warn("Select an object in the object list first")
            return
        label, bbox_min, bbox_max, image_layer = selected.label, selected.bbox_min, selected.bbox_max, selected.image_layer
        if image_layer.ndim != 3 or image_layer.rgb:
            warnings.warn("Suggesting anchors needs a 3D image layer")
            return
        if self.used_feature == CUSTOM:
            show_info("The suggestion uses the gradient feature, not the custom one")
        result = suggest_anchors_in_image(
            image_layer.data, bbox_min, bbox_max, resample=(self.z_scale, 1., 1.), alpha=self.alpha, beta=self.beta,
            blur_func=self._blur_image if self.is_blurring_enabled else None,
            **self._suggestion_options()
        )
        if not result.found:
            warnings.warn("No anchor suggestion for object %d: %s" % (label, result.reason))
            return
        anchor_store(selected.bbox_layer).set(label, [result.p1, result.p2])
        self._update_object_state()
        message = "Suggested anchors for object %d (confidence %.2f%s)" % (
            label, result.confidence, ", blurred" if self.is_blurring_enabled else ", not blurred"
        )
        if result.confidence < .2:
            message += ", low: check the points"
        show_info(message)

    def _suggestion_options(self):
        method = SUGGESTION_MODES[self.suggestion_mode_combobox.value]
        options = {"method": method}
        if method == "fan":
            options.update(
                n_rays=self.fan_rays_spinbox.value,
                max_radius_jump=self.fan_max_radius_jump_spinbox.value,
                min_relative_strength=self.fan_min_relative_strength_spinbox.value,
            )
        return options

    @suggestion_mode_combobox.connect
    def _on_suggestion_mode_changed(self, mode):
        is_fan = SUGGESTION_MODES[mode] == "fan"
        for widget in (self.fan_rays_spinbox, self.fan_max_radius_jump_spinbox,
                       self.fan_min_relative_strength_spinbox):
            widget.visible = is_fan

    def _selected_object(self):
        return selected_object_info(self._viewer) if self._viewer is not None else None

    def _anchor_pair(self):
        """The anchors of the selected object as two points in data coordinates of its image layer, or None."""
        selected = self._selected_object()
        if selected is None:
            return None
        anchors = anchor_store(selected.bbox_layer).get(selected.label)
        return (anchors[0], anchors[1]) if len(anchors) == 2 else None

    def _start_estimation(self):
        selected = self._selected_object()
        if selected is None:
            warnings.warn("No object selected: select one in the object list")
            return
        image, labels = selected.image_layer, selected.labels_layer
        if image.ndim != 3 or image.rgb:
            warnings.warn("Minimal Surface works only with 3D, single channel images")
            return
        if labels is None:
            warnings.warn("Select a labels layer in the object list")
            return
        if not np.array_equal(labels.data.shape[:labels.ndim], image.data.shape[:image.ndim]):
            warnings.warn("The labels layer of the object list and its image have different shapes")
            return
        if np.issubdtype(labels.dtype, np.integer) and selected.label > np.iinfo(labels.dtype).max:
            warnings.warn("The label id %d does not fit the data type of the labels layer (%s)" % (selected.label, labels.dtype))
            return
        anchors = anchor_store(selected.bbox_layer).get(selected.label)
        if len(anchors) < 2:
            warnings.warn("Object %d needs two anchors: click on its surface in the 3D view or press 'Suggest anchors'"
                          % selected.label)
            return
        self._run_target = (labels, selected.label, (selected.bbox_min, selected.bbox_max))
        image_data = image.data
        if self.slice_annotation_method in EstimationWorker.AUTOMATIC_METHODS and self.used_feature == GRADIENT:
            warnings.warn("The automatic slice annotation needs a feature that tells the object from the background:"
                          " choose 'Gradient + intensity' (or a custom feature) under Image Features")
            return
        phi_func = {GRADIENT: None, GRADIENT_INTENSITY: gradient_intensity_features}.get(
            self.used_feature, self._run_feature_extractor)
        points_data = anchors.copy()
        z_scale = self.z_scale
        alpha = self.alpha
        beta = self.beta
        n_iter = self.iterations
        use_gradient = self.used_feature == GRADIENT
        use_meeting_plane_points = True
        estimation_worker = EstimationWorker(self._viewer, self.minimal_contour_widget, self)
        estimation_worker.initialize(image_data, phi_func, points_data, beta, alpha, n_iter, use_meeting_plane_points, self.slice_annotation_method, self._blur_image if self.is_blurring_enabled else None, z_scale, use_gradient)
        estimation_worker.image_data_received.connect(self._add_image)
        estimation_worker.layer_invalidated.connect(self._refresh_layer)
        estimation_worker.mask_data_received.connect(self._mask_received)
        estimation_worker.distance_map_received.connect(self._distance_map_received)
        estimation_worker.contour_points_received.connect(self._contour_points_received)
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

    @magicclass(widget_type="collapsible", name="Shortest Paths")
    class ShortestPathsWidget(MagicTemplate):
        ...

    used_feature_combobox = field(widget_type="ComboBox", location=ImageFeaturesWidget).with_choices(FEATURE_KEYS)
    feature_editor = field(ScriptExecuteWidget, location=ImageFeaturesWidget).with_options(editor_key="minimal_surface_features")
    alpha = vfield(float, label="\u03B1", widget_type="FloatSlider", location=ImageFeaturesWidget).with_options(min=0, max=1, value=0.01)
    beta = vfield(float, label="\u03B2", location=ImageFeaturesWidget).with_options(min=.01, max=500., value=5)
    iterations = vfield(int, widget_type="Slider", location=ImageFeaturesWidget).with_options(min=1, max=10000, value=10000)
    z_scale = vfield(float, location=ImageFeaturesWidget).with_options(min=0.01, max=20., value=1.)

    distance_map_combobox = field(Image, label="Distance map (anchor 1)", location=ShortestPathsWidget)
    second_distance_map_combobox = field(Image, label="Distance map (anchor 2)", location=ShortestPathsWidget)
    start_points_combobox = field(Points, label="Start points", location=ShortestPathsWidget)

    blur_image_checkbox = field(bool, label="Blur image", location=BlurringWidget)
    blurring_type_combobox = (field(str, widget_type="ComboBox", location=BlurringWidget)
                              .with_choices([GAUSSIAN_BLURRING, CAD_BLURRING]))
    blur_sigma_slider = field(float, widget_type="FloatSlider", location=BlurringWidget).with_options(max=20)
    conductance_spinbox = field(float, location=BlurringWidget).with_options(max=100, value=9)
    cad_iterations_spinbox = field(int, label="Iterations", location=BlurringWidget).with_options(min=1, max=20, value=5)

    slice_annotation_method = vfield(str, location=SliceAnnotationWidget).with_options(
        choices=annotation_methods(),
        tooltip="How the contour of the object in the plane where the fronts of the two anchors meet is found."
                " Manual, Threshold, Minimal contour: you annotate it. Automatic (distances): from the distances of"
                " the anchors in the plane, which is instant (falls back to the phase field if they give nothing"
                " usable). Automatic (phase field): evolves a phase field in the plane, which takes seconds."
    )

    def __init__(self):
        self._collapsible_group = CollapsibleContainerGroup()
        self._viewer = None
        self._run_target = None
        self._anchors_layer = None
        self._plane_image_layer = None
        self.blurred_layer = None
        self._try_blurring_button = None
        self.blurring_progress_widget = ProgressWidget(message="blurring...")
        self.estimation_progress_widget = ProgressWidget(message="Annotating...")
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

    def __post_init__(self):
        self.call_button.clicked.connect(self._start_estimation)
        self.suggest_anchors_button.clicked.connect(self._suggest_anchors)
        self.clear_anchors_button.clicked.connect(self._clear_anchors)
        self.native.setMinimumWidth(self.native.children()[1].widget().sizeHint().width()
                                    + self.native.children()[1].verticalScrollBar().sizeHint().width()
                                    + 2 * self.native.children()[1].widget().layout().getContentsMargins()[2])
        for container in [self.ImageFeaturesWidget, self.BlurringWidget, self.SliceAnnotationWidget, self.ShortestPathsWidget]:
            container._widget._expand_btn.setChecked(False)
            # container._widget._inner_qwidget.setStyleSheet("padding: 0 10 0 10")
            container.collapsed = True
            self._collapsible_group.addItem(container)
        self._on_feature_changed(self.used_feature)
        self._on_suggestion_mode_changed(self.suggestion_mode_combobox.value)
        self.try_blurring_button.native.setCheckable(True)
        self._on_blur_checked(self.blur_image_checkbox.value)

        self._ensure_3d_image()

    # widgets that hold a layer: layers are not serializable
    _NOT_SERIALIZED = ("image_layer_combobox", "object_status_label", "distance_map_combobox",
                       "second_distance_map_combobox", "start_points_combobox")

    def __magicclass_serialize__(self):
        def remove_layer_widgets(d):
            # the widgets in collapsible containers (e.g. "Shortest Paths") are nested
            for key in list(d):
                if key in self._NOT_SERIALIZED:
                    del d[key]
                elif isinstance(d[key], dict):
                    remove_layer_widgets(d[key])
            return d

        return remove_layer_widgets(serialize(self))

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
        self._viewer = unwrap_viewer(viewer)
        self.minimal_contour_widget = minimal_contour_widget
        self._viewer.mouse_drag_callbacks.append(self._on_canvas_mouse)
        selected_object_changed.connect(self._on_selected_object_changed)
        self._ensure_3d_image()
        self._update_custom_feature_image()
        self._update_object_state()

    @set_design(text="Try", location=BlurringWidget,)
    def try_blurring(self):
        self._toggle_blur_demo()

    @slice_annotation_method.connect
    def _on_annotation_method_changed(self, method):
        if method in EstimationWorker.AUTOMATIC_METHODS and self.used_feature == GRADIENT:
            self.used_feature_combobox.value = GRADIENT_INTENSITY

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
        if self._run_target is None:
            return
        labels_layer, label, bbox = self._run_target
        write_object_mask(labels_layer, label, data, offset, bbox)
        if labels_layer in self._viewer.layers:
            # the layers added during the run took the selection; Ctrl+Z undoes the result on the active labels layer
            self._viewer.layers.selection.select_only(labels_layer)

    def _distance_map_received(self, layer_name, anchor_idx, data, layer_args):
        if layer_name in self._viewer.layers:
            layer = self._viewer.layers[layer_name]
            layer.data = data
            layer.translate = layer_args["translate"]
            layer.scale = layer_args["scale"]
        else:
            layer = self._viewer.add_image(data, name=layer_name, colormap="plasma", visible=False, **layer_args)
        [self.distance_map_combobox, self.second_distance_map_combobox][anchor_idx].value = layer

    def _contour_points_received(self, layer_name, points, layer_args):
        if points is None:
            return
        if layer_name in self._viewer.layers:
            layer = self._viewer.layers[layer_name]
            layer.data = points
            layer.translate = layer_args["translate"]
            layer.scale = layer_args["scale"]
        else:
            layer = self._viewer.add_points(points, name=layer_name, ndim=3, size=2, **layer_args)
        self.start_points_combobox.value = layer

    @set_design(text="Show shortest paths", location=ShortestPathsWidget)
    def show_shortest_paths(self):
        points_layer = self.start_points_combobox.value
        distance_map_layers = [self.distance_map_combobox.value, self.second_distance_map_combobox.value]
        if points_layer is None or all(layer is None for layer in distance_map_layers):
            warnings.warn("Select a distance map and a Points layer with the start points")
            return
        if points_layer.ndim != 3 or any(layer is not None and layer.ndim != 3 for layer in distance_map_layers):
            warnings.warn("The distance maps and the start points should be 3D")
            return
        if len(points_layer.data) == 0:
            warnings.warn("The start points layer is empty")
            return
        estimator = minimal_surface.MinimalSurfaceCalculator()
        colors = np.asarray(generate_label_colors(len(points_layer.data))) / 255
        # every start point is traced on the distance map of each anchor, so the path leads to that anchor
        for anchor_idx, distance_map_layer in enumerate(distance_map_layers):
            layer_name = "Shortest paths (anchor %d)" % (anchor_idx + 1)
            if layer_name in self._viewer.layers:
                self._viewer.layers.remove(layer_name)
            if distance_map_layer is None:
                continue
            # data_to_world and world_to_data convert a single position, not an array of them
            points = [distance_map_layer.world_to_data(points_layer.data_to_world(point)) for point in points_layer.data]
            points = np.rint(points).astype(np.int64)
            inside = np.all((points >= 0) & (points < np.asarray(distance_map_layer.data.shape)), axis=1)
            if not inside.all():
                warnings.warn("Skipped %d start point(s) outside '%s'" % ((~inside).sum(), distance_map_layer.name))
            distance_map = np.ascontiguousarray(distance_map_layer.data, dtype=float)
            # a trace that starts on the outermost voxel layer of the map stalls there, so it starts one voxel inside
            starts = np.clip(points, 1, np.asarray(distance_map.shape) - 2)
            paths = []
            path_colors = []
            for idx in np.flatnonzero(inside):
                path = np.vstack([points[idx], estimator.resolve_shortest_paths(starts[idx], distance_map)])
                # a path ends by repeating its last point up to the iteration limit
                path = path[np.r_[True, np.any(np.diff(path, axis=0) != 0, axis=1)]]
                if len(path) > 1:
                    paths.append(path)
                    path_colors.append(colors[idx])
            if not paths:
                warnings.warn("No paths on '%s': the start points are outside it or already at the source"
                              % distance_map_layer.name)
                continue
            self._viewer.add_layer(Shapes(
                paths,
                name=layer_name,
                shape_type="path",
                edge_color=np.concatenate([path_colors, np.ones([len(paths), 1])], axis=1),
                edge_width=1,
                scale=distance_map_layer.scale,
                translate=distance_map_layer.translate,
                rotate=distance_map_layer.rotate,
                shear=distance_map_layer.shear,
                affine=distance_map_layer.affine
            ))

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
        pair = self._anchor_pair()
        image_layer = self.image_layer
        if pair is None or image_layer is None:
            return
        if is_checked:
            bb = pts_2_bb(pair[0], pair[1], image_layer.data.shape, image_layer.scale)
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
        pair = self._anchor_pair()
        if pair is None:
            return
        bb = pts_2_bb(pair[0], pair[1], image_layer.data.shape, image_layer.scale)
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
        for container in self._collapsible_group:
            container.enabled = is_enabled
        self.call_button.enabled = is_enabled
        self.suggest_anchors_button.enabled = is_enabled
        self.clear_anchors_button.enabled = is_enabled
        self.place_anchors_checkbox.enabled = is_enabled
        self.anchor_placement_combobox.enabled = is_enabled
        self.suggestion_mode_combobox.enabled = is_enabled
        for widget in (self.fan_rays_spinbox, self.fan_max_radius_jump_spinbox,
                       self.fan_min_relative_strength_spinbox):
            widget.enabled = is_enabled

    @image_layer_combobox.connect
    def _update_custom_feature_image(self, _=None):
        if self.image_layer is not None and self.image_layer.ndim != 2:
            self.feature_editor.variables["image"] = self.image_layer.data

    # --- objects and their anchors

    def _on_selected_object_changed(self, event):
        if getattr(event, "viewer", None) is not self._viewer:
            return
        self._update_object_state()

    def _update_object_state(self):
        """Follows the selected object: the image, the anchors that are shown, the status and the plane."""
        selected = self._selected_object()
        if selected is None:
            self.object_status_label.value = "none selected"
            self._show_anchors(None, np.zeros((0, 3)), "")
            self._apply_placement_mode()
            return
        image_layer = selected.image_layer
        if image_layer.ndim == 3 and not image_layer.rgb and image_layer is not self.image_layer:
            self.image_layer = image_layer
        anchors = anchor_store(selected.bbox_layer).get(selected.label)
        self.object_status_label.value = "%d (%d/2 anchors)" % (selected.label, len(anchors))
        world = np.asarray([image_layer.data_to_world(point) for point in anchors]).reshape(-1, 3) \
            if image_layer.ndim == 3 else np.zeros((0, 3))
        self._show_anchors(image_layer, world, "Anchors of object %d [display only]" % selected.label)
        self._apply_placement_mode(selected)

    def _show_anchors(self, image_layer, world_points, name):
        """Shows the anchors of the selected object on a Points layer that the plugin owns."""
        layer = self._anchors_layer
        if layer is not None and layer not in self._viewer.layers:
            layer = self._anchors_layer = None
        if layer is None:
            if len(world_points) == 0:
                return
            selection = self._viewer.layers.selection
            previous, active = set(selection), selection.active
            layer = Points(
                world_points, ndim=3, name=name, size=self._anchor_size(image_layer),
                face_color=ANCHOR_COLORS[:len(world_points)], metadata={SCAFFOLD_KEY: True}
            )
            self._viewer.add_layer(layer)
            layer.editable = False
            # adding a layer selects it; the layers of the object (labels, image) stay the ones that are selected
            selection.clear()
            selection.update(previous)
            if active in previous:
                selection.active = active
            self._viewer.layers.events.inserted.connect(keep_layer_on_top(layer))
            self._viewer.layers.events.moved.connect(keep_layer_on_top(layer))
            self._anchors_layer = layer
            return
        layer.visible = len(world_points) > 0
        if len(world_points) == 0:
            return
        layer.name = name
        layer.size = self._anchor_size(image_layer)
        layer.data = world_points
        layer.face_color = ANCHOR_COLORS[:len(world_points)]
        layer.editable = False

    @staticmethod
    def _anchor_size(image_layer):
        if image_layer is None:
            return 2
        extent = np.asarray(image_layer.data.shape[:3]) * np.asarray(image_layer.scale)[-3:]
        return max(2., .02 * extent.max())

    def _clear_anchors(self):
        selected = self._selected_object()
        if selected is None:
            warnings.warn("Select an object in the object list first")
            return
        anchor_store(selected.bbox_layer).clear(selected.label)
        self._update_object_state()

    @anchor_placement_combobox.connect
    def _on_placement_changed(self, mode=None):
        self._apply_placement_mode()

    def _apply_placement_mode(self, selected=None):
        """Shows the image of the selected object with the plane depiction in the "Image plane" placement mode."""
        selected = selected or self._selected_object()
        target = None
        if selected is not None and self.anchor_placement_combobox.value == IMAGE_PLANE and selected.image_layer.ndim == 3:
            target = selected.image_layer
        previous, self._plane_image_layer = self._plane_image_layer, target
        if previous is not None and previous is not target:
            try:
                previous.depiction = "volume"
            except (AttributeError, ValueError, RuntimeError):  # the layer was removed
                pass
        if target is None:
            return
        if target.depiction != "plane":
            target.depiction = "plane"
        center = (selected.bbox_min + selected.bbox_max) / 2
        position = np.asarray(target.plane.position)
        if np.any(position < selected.bbox_min - .5) or np.any(position > selected.bbox_max + .5):
            target.plane.position = tuple(float(c) for c in center)

    def _placing_anchors(self, event):
        """The selected object if a click like ``event`` places its anchors, else None."""
        if not self.place_anchors_checkbox.value or self._viewer.dims.ndisplay != 3:
            return None
        if event.type != "mouse_press" or event.button not in (1, 2) or len(event.modifiers) > 0:
            return None
        try:
            if not self.native.isVisibleTo(self.native.window()):  # another tab is shown, or the toolbox is closed
                return None
        except RuntimeError:  # the widget was deleted
            return None
        selected = self._selected_object()
        if selected is None or selected.image_layer.ndim != 3 or selected.image_layer.rgb:
            return None
        return selected

    def _on_canvas_mouse(self, viewer, event):
        if self._placing_anchors(event) is None:
            return
        button = event.button
        position, view_direction, dims_displayed = event.position, event.view_direction, event.dims_displayed
        dragging = False
        yield
        while event.type == "mouse_move":  # rotating the view
            dragging = True
            yield
        if not dragging:
            self.place_anchor(button, position, view_direction, dims_displayed)

    def place_anchor(self, button, position, view_direction, dims_displayed):
        """A click in the 3D view: the left button places an anchor of the selected object, the right removes one."""
        selected = self._selected_object()
        if selected is None:
            show_info("Select an object in the object list first")
            return
        store = anchor_store(selected.bbox_layer)
        if button == 2:
            store.pop_last(selected.label)
        else:
            point = self._pick_anchor(selected, position, view_direction, dims_displayed)
            if point is None:
                show_info("No anchor here: the ray does not hit the %s inside the bounding box of object %d"
                          % ("image plane" if self.anchor_placement_combobox.value == IMAGE_PLANE else "image data",
                             selected.label))
                return
            store.add(selected.label, point)
        self._update_object_state()

    def _pick_anchor(self, selected, position, view_direction, dims_displayed):
        """The anchor under the cursor ray (data coordinates of the image of the object), or None."""
        image_layer = selected.image_layer
        start, end = image_layer.get_ray_intersections(
            position=position, view_direction=view_direction, dims_displayed=list(dims_displayed), world=True)
        if start is None or end is None:
            return None
        if self.anchor_placement_combobox.value == IMAGE_PLANE and image_layer.depiction == "plane":
            return pick_on_plane(start, end, image_layer.plane.position, image_layer.plane.normal,
                                 selected.bbox_min, selected.bbox_max)
        return pick_along_ray(image_layer.data, start, end, selected.bbox_min, selected.bbox_max)

if minimal_surface is not None:
    import SimpleITK as sitk
    MinimalSurfaceWidget = _MinimalSurfaceWidget
else:
    MinimalSurfaceWidget = None

