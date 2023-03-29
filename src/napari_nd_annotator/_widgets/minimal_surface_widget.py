from time import sleep

import tifffile
from napari.utils.notifications import show_info

from ._utils import WidgetWithLayerList, CollapsibleWidget, CollapsibleWidgetGroup, ProgressWidget
from ._utils.delayed_executor import DelayedExecutor
from ._utils.callbacks import keep_layer_on_top
import napari
from napari.layers import Image, Labels, Points, Shapes
from napari.utils.colormaps.standardize_color import transform_color
from napari._qt.widgets._slider_compat import QDoubleSlider
from qtpy.QtCore import Signal, Qt, QObject, QThread
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QFrame,
    QWidget,
    QApplication,
    QComboBox,
    QSpinBox,
    QSizePolicy
)
from threading import Thread
import cv2
import numpy as np
import skimage.measure, skimage.morphology
from scipy.ndimage import gaussian_filter, zoom
import itertools
import colorsys
from random import shuffle, seed
import SimpleITK as sitk

try:
    import MinArea
except ImportError:
    MinArea = None


def pts_2_bb(p1, p2, image_size, scale=1.):
    center = (p1 + p2) / 2
    size = np.sqrt(((p1 - p2) ** 2).sum()) + 10
    size = np.divide(size, scale)
    bb = np.asarray(np.where(list(itertools.product((False, True), repeat=3)), center + size / 2, center - size / 2))
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
    r = max(shape)//2
    mask = skimage.morphology.ball(r)
    shape_diffs = tuple(shape[i] - mask.shape[i] for i in range(ndim))
    crop_idx = tuple(slice(max(0, round(-shape_diffs[i]/2+0.1)), mask.shape[i]+min(0, round(shape_diffs[i]/2+0.1))) for i in range(ndim))
    mask = mask[crop_idx]
    mask = np.pad(mask, [(max(0, round(shape_diffs[i]/2)), -min(0, round(-shape_diffs[i]/2+0.1))) for i in range(ndim)])
    assert mask.shape == shape
    return mask


if MinArea is not None:
    class ColorPairsCallback:
        def __init__(self, n_colors=50):
            self.prev_selection = set()
            self.prev_len = 0
            self.prev_data = None
            color_cycle = list(
                map(lambda color_tuple: color_to_hex_string(color_tuple), generate_label_colors(n_colors)))
            self.color_cycle = itertools.cycle(color_cycle)

        def __call__(self, event):
            points_layer = event.source
            if len(points_layer.selected_data) > 1:
                points_layer.selected_data = self.prev_selection
                points_layer.refresh()
                return
            self.prev_selection = points_layer.selected_data.copy()
            if event.type == "data":
                if len(points_layer.data) > self.prev_len:
                    if len(points_layer.data) % 2 == 1:
                        points_layer.current_face_color = next(self.color_cycle)
                        points_layer.face_color[-1] = transform_color(points_layer.current_face_color)
                    self.prev_data = points_layer.data.copy()
                else:
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
        MANUAL_ANNOTATION = "Manual"
        THRESHOLD_ANNOTATION = "Threshold"
        MINIMAL_CONTOUR_ANNOTATION = "Minimal contour"
        ANNOTATION_METHODS = [MANUAL_ANNOTATION, THRESHOLD_ANNOTATION, MINIMAL_CONTOUR_ANNOTATION]

        def __init__(self, viewer, minimal_contour_widget, parent=None):
            super().__init__(parent)
            self.viewer = viewer
            self.minimal_contour_widget = minimal_contour_widget
            self.center_mat = None
            self.rotation_matrix = None
            self.translation_matrix = None
            self.image = None
            self.points = None
            self.use_correction = None
            self.beta = None
            self.alpha = None
            self.done_pressed = False
            self.blur_func = None
            self.init_slice_annotation_fun = None
            self.slice_image_layer = None
            self.slice_labels_layer = None
            self.slice_annotation = None
            self.manual_annotation_done = False
            self.z_scale = None
            #create annotation dialogs
            self.manual_annotation_dialog = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Annotate image, then click 'Done'"))
            annotation_done_button = QPushButton("Done")
            layout.addWidget(annotation_done_button)
            def on_done_pressed():
                self.done_pressed = True
            annotation_done_button.clicked.connect(self.finish_manual_annotation)
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
            annotation_done_button.clicked.connect(self.finish_threshold_annotation)
            annotation_done_button.clicked.connect(on_done_pressed)
            self.threshold_annotation_dialog.setLayout(layout)
            self.annotation_needed_signal.connect(self.begin_slice_annotation)
            self.slice_annotation_methods = {EstimationWorker.MANUAL_ANNOTATION: self.manual_annotation, EstimationWorker.THRESHOLD_ANNOTATION: self.threshold_annotation, EstimationWorker.MINIMAL_CONTOUR_ANNOTATION: self.minimal_contour_annotation}
        
        def begin_slice_annotation(self, image_slice, distance_map_slice):
            self.slice_image_layer = self.viewer.add_image(image_slice)
            self.slice_labels_layer = self.viewer.add_labels(np.zeros_like(image_slice, dtype=int))
            self.slice_labels_layer.brush_size = 1
            self.slice_labels_layer.mode = "paint"
            self.init_slice_annotation_fun()
        
        def finish_slice_annotation(self):
            self.slice_annotation = self.slice_labels_layer.data > 0
            self.viewer.layers.remove(self.slice_image_layer)
            self.viewer.layers.remove(self.slice_labels_layer)
            self.slice_labels_layer = None
            self.slice_image_layer = None

        def manual_annotation(self):
            self.manual_annotation_dialog.show()
        
        def finish_manual_annotation(self):
            self.manual_annotation_dialog.hide()
            self.finish_slice_annotation()
        
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
            self.finish_slice_annotation()
        
        def minimal_contour_annotation(self):
            self.viewer.layers.selection.select_only(self.minimal_contour_widget.anchor_points)
            self.minimal_contour_widget.image.layer = self.slice_image_layer
            self.minimal_contour_widget.labels.layer = self.slice_labels_layer
            self.manual_annotation_dialog.show()
        
        def finish_minimal_contour_annotation(self):
            self.manual_annotation_dialog.hide()
            self.finish_slice_annotation()

        def initialize(self, image, points, use_correction, beta, alpha, slice_annotation_method="Manual", blur_func=None, z_scale=1.):
            self.image = image
            self.points = points
            self.use_correction = use_correction
            self.beta = beta
            self.alpha = alpha
            self.blur_func = blur_func
            self.init_slice_annotation_fun = self.slice_annotation_methods[slice_annotation_method]
            self.z_scale = z_scale

        def run(self):
            import time
            start = time.time()
            print("started")
            postscript = ""
            estimator = MinArea.MinimalSurfaceCalculator()
            # estimator.hook_stage_data_init_event(
            #     MinArea.AREA_EIKONAL_STAGE,
            #     lambda arr, idx: self.data_initializer(
            #         "Area Eikonal %d%s" % (0, postscript),
            #         0,
            #         {
            #             'colormap': "plasma",
            #             'translate': offset,
            #             "visible": False
            #         }
            #     )(arr, idx)
            # )
            # estimator.hook_stage_iteration_event(MinArea.AREA_EIKONAL_STAGE,
            #                                      lambda idx: self.data_updater("Area Eikonal %d%s" % (0, postscript))(
            #                                          idx))
            # estimator.hook_stage_finished_event(MinArea.AREA_EIKONAL_STAGE,
            #                                     lambda: self.data_finalizer("Area Eikonal %d%s" % (0, postscript))())
            # estimator.hook_stage_data_init_event(
            #     MinArea.AREA_EIKONAL_STAGE,
            #     lambda arr, idx: self.data_initializer(
            #         "Area Eikonal %d%s" % (1, postscript),
            #         1,
            #         {
            #             'colormap': "plasma",
            #             'translate': offset,
            #             "visible": False
            #         }
            #     )(arr, idx)
            # )
            # estimator.hook_stage_iteration_event(MinArea.AREA_EIKONAL_STAGE,
            #                                      lambda idx: self.data_updater("Area Eikonal %d%s" % (1, postscript))(
            #                                          idx))
            # estimator.hook_stage_finished_event(MinArea.AREA_EIKONAL_STAGE,
            #                                     lambda: self.data_finalizer("Area Eikonal %d%s" % (1, postscript))())
            # self.hook_callbacks(estimator, MinArea.AREA_EIKONAL_STAGE, "Area Eikonal 0", {'colormap': "plasma", 'translate': offset, "visible": False}, 0)
            # self.hook_callbacks(estimator, MinArea.AREA_EIKONAL_STAGE, "Area Eikonal 1", {'colormap': "plasma", 'translate': offset, "visible": False}, 1)
            # self.hook_callbacks(estimator, MinArea.ROTATED_AREA_EIKONAL_STAGE, "Rotated Area Eikonal", {'colormap': "plasma"}, 0)

            # estimator.hook_stage_data_init_event(
            #     MinArea.PLANE_PHASEFIELD_STAGE,
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
            # estimator.hook_stage_iteration_event(MinArea.PLANE_PHASEFIELD_STAGE,
            #                                      lambda idx: self.data_updater("Plane PhaseField%s" % postscript)(idx))
            # estimator.hook_stage_finished_event(MinArea.PLANE_PHASEFIELD_STAGE,
            #                                     lambda: self.data_finalizer("Plane PhaseField%s" % postscript)())

            # estimator.hook_stage_data_init_event(
            #     MinArea.TRANSPORT_FUNCTION_STAGE,
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
            # # estimator.hook_stage_data_init_event(MinArea.TRANSPORT_FUNCTION_STAGE, lambda arr, idx: print("tf init"))
            # estimator.hook_stage_iteration_event(MinArea.TRANSPORT_FUNCTION_STAGE,
            #                                      lambda idx: self.data_updater("Transport Function%s" % postscript)(
            #                                          idx))
            # # estimator.hook_stage_iteration_event(MinArea.TRANSPORT_FUNCTION_STAGE, lambda idx: print("tf iter %d" % idx))
            # estimator.hook_stage_finished_event(MinArea.TRANSPORT_FUNCTION_STAGE,
            #                                     lambda: self.data_finalizer("Transport Function%s" % postscript)())
            #
            def tform_calculated(rotation, translation):
                self.rotation_matrix = np.flip(rotation.copy()).reshape(3, 3)
                self.translation_matrix = np.flip(translation.copy())

            estimator.hook_transform_calculated_event(tform_calculated)

            # def center_calculated(center):
            #     self.center_mat = np.flip(center.copy())
            #
            # estimator.hook_plane_center_calculated_event(center_calculated)
            estimator.set_initial_plane_calculator(self.segment_initial_slice)
            print("callbacks attached in %d seconds" % (time.time() - start))
            timestamp = time.time()
            for i in range(len(self.points) // 2):
                bounding_box = pts_2_bb(self.points[2 * i], self.points[2 * i + 1], self.image.shape, scale=(self.z_scale, 1, 1))
                bounding_box = np.clip(bounding_box, 0, np.asarray(self.image.shape) - 1).round().astype(int)
                offset = bounding_box.min(0, keepdims=True)
                bb_slice = bb_2_slice(bounding_box)
                point1 = np.flip(self.points[2 * i] - offset).reshape(-1)
                point2 = np.flip(self.points[2 * i + 1] - offset).reshape(-1)
                point1[2] *= self.z_scale
                point2[2] *= self.z_scale
                print(point1, point2)
                data = self.image[bb_slice]
                data = zoom(data, (self.z_scale, 1, 1))
                data = (data - data.min()) / (data.max() - data.min())
                # data = scipy.ndimage.gaussian_filter(data, self.blur_sigma)
                if self.blur_func is not None:
                    data = self.blur_func(data)
                offset = np.clip(bounding_box.min(0), 0, np.asarray(self.image.shape) - 1)
                # start = time.time()

                output = estimator.calculate(data, point1, point2, self.use_correction, self.beta, self.alpha, 1000)
                tifffile.imwrite("C:\\Users\\Tejfel\\Downloads\\tf.tiff", output)
                tifffile.imwrite("C:\\Users\\Tejfel\\Downloads\\zoomed.tiff", data)
                segmented = (output >= 0)
                labelled = skimage.measure.label(segmented)
                obj_pixel = np.argwhere(output == output.max())[0]
                obj_label = labelled[tuple(obj_pixel)]
                mask = (labelled == obj_label) * (i + 1)
                mask = zoom(mask, (1/self.z_scale, 1, 1), order=0)
                # self.image_data_received.emit("Result", output, {"colormap": "plasma", "translate": offset})
                self.mask_data_received.emit(mask, offset)
                postscript = " - %d" % (i + 1)
                # self.remove_layer.emit("Transport Function")
            show_info("Estimation done in %d" % (time.time()-start))
            self.all_done.emit()

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
            def update_viewer(iteration):
                it[0] += 1
                if it[0] % 100 == 0: #TODO undo
                    self.layer_invalidated.emit(name)
                    tifffile.imwrite("C:\\Users\\Tejfel\\Downloads\\temp_tf_%.7d.tif" % it[0], self.viewer.layers[name].data)
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
            while not self.done_pressed:
                sleep(0.1)
                QApplication.processEvents()
            self.done_pressed = False
            mask = self.slice_annotation.astype(float)
            self.slice_annotation = None
            return mask

    class MinimalSurfaceWidget(WidgetWithLayerList):
        def __init__(self, viewer: napari.Viewer, minimal_contour_widget):
            super().__init__(viewer, [("image", Image), ("labels", Labels)])
            self.blurred_layer = None
            self.progress_widget = ProgressWidget(message="blurring...")
            self.minimal_contour_widget = minimal_contour_widget
            self.points_layer = self.viewer.add_points(ndim=3, size=2, name="3D Anchors [DO NOT REMOVE]")
            self.points_layer.events.connect(ColorPairsCallback())
            self.points_layer.mouse_drag_callbacks.append(self.drag_points_callback)
            self.viewer.layers.events.connect(keep_layer_on_top(self.points_layer))
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignTop)
            # Parameters for the estimator
            params_widget = CollapsibleWidget("Image features")
            params_widget.setContentsMargins(0, 0, 0, 0)
            params_layout = QVBoxLayout()
            self.image_feature_combobox = QComboBox(self)
            self.image_feature_combobox.addItem("Gradient")
            params_layout.addWidget(self.image_feature_combobox)

            alpha_layout = QHBoxLayout()
            alpha_layout.addWidget(QLabel("\u03B1", parent=self))
            self.alpha_slider = QDoubleSlider(parent=self)
            self.alpha_slider.setMinimum(0.)
            self.alpha_slider.setMaximum(1.)
            self.alpha_slider.setValue(0.01)
            self.alpha_slider.setOrientation(Qt.Horizontal)
            alpha_layout.addWidget(self.alpha_slider)
            params_layout.addLayout(alpha_layout)

            beta_layout = QHBoxLayout()
            beta_layout.addWidget(QLabel("\u03B2", parent=self))
            self.beta_spinner = QDoubleSpinBox()
            self.beta_spinner.setMinimum(1.)
            self.beta_spinner.setMaximum(100.)
            self.beta_spinner.setValue(30.)
            self.beta_spinner.setSingleStep(1.)
            self.beta_spinner.setSizePolicy(QSizePolicy.Policy.Expanding, self.beta_spinner.sizePolicy().verticalPolicy())
            beta_layout.addWidget(self.beta_spinner)
            params_layout.addLayout(beta_layout)

            z_scale_layout = QHBoxLayout()
            z_scale_layout.addWidget(QLabel("Z-scale"))
            self.z_scale_spinbox = QDoubleSpinBox()
            self.z_scale_spinbox.setMinimum(0.01)
            self.z_scale_spinbox.setMaximum(20)
            self.z_scale_spinbox.setSingleStep(1.)
            self.z_scale_spinbox.setValue(1.)
            z_scale_layout.addWidget(self.z_scale_spinbox)
            params_layout.addLayout(z_scale_layout)

            self.use_correction_checkbox = QCheckBox("Use correction")
            self.use_correction_checkbox.setChecked(True)
            params_layout.addWidget(self.use_correction_checkbox)

            params_widget.setLayout(params_layout)
            layout.addWidget(params_widget, 0, Qt.AlignTop)

            # Blurring parameters
            blurring_widget = CollapsibleWidget("Blurring", self)
            blurring_layout = QVBoxLayout()
            self.blur_image_checkbox = QCheckBox("Blur image")
            self.blur_image_checkbox.clicked.connect(self.on_blur_checked)
            blurring_layout.addWidget(self.blur_image_checkbox)
            self.blurring_type_widget = QWidget()
            blurring_type_layout = QVBoxLayout()
            blurring_type_layout.setContentsMargins(0, 0, 0, 0)
            blurring_type_layout.addWidget(QLabel("Blurring type", self))
            self.blurring_type_combobox = QComboBox()
            self.blurring_type_combobox.addItems(["Gaussian", "Curvature Anisotropic Diffusion"])
            self.blurring_type_combobox.setSizePolicy(QSizePolicy.Policy.Ignored, self.blurring_type_combobox.sizePolicy().verticalPolicy())
            self.blurring_type_combobox.currentTextChanged.connect(self.update_blurred_layer)
            blurring_type_layout.addWidget(self.blurring_type_combobox)
            self.blurring_type_widget.setLayout(blurring_type_layout)
            blurring_layout.addWidget(self.blurring_type_widget)
            self.cad_widget = QWidget()
            cad_layout = QVBoxLayout()
            cad_layout.setContentsMargins(0, 0, 0, 0)
            conductance_layout = QHBoxLayout()
            conductance_layout.addWidget(QLabel("Conductance"))
            delayed_updater = DelayedExecutor(self.update_blurred_layer)
            delayed_updater.processing.connect(lambda _: self.progress_widget.setVisible(blur_demo_btn.isChecked()))
            delayed_updater.processed.connect(lambda _: self.progress_widget.setVisible(False))
            self.blur_conductance_spinbox = QDoubleSpinBox()
            self.blur_conductance_spinbox.setMinimum(0)
            self.blur_conductance_spinbox.setMaximum(100)
            self.blur_conductance_spinbox.setValue(9.)
            self.blur_conductance_spinbox.valueChanged.connect(delayed_updater)
            conductance_layout.addWidget(self.blur_conductance_spinbox)
            cad_layout.addLayout(conductance_layout)

            iterations_layout = QHBoxLayout()
            iterations_layout.addWidget(QLabel("# iterations"))
            self.blur_n_iterations_spinbox = QSpinBox()
            self.blur_n_iterations_spinbox.setMinimum(1)
            self.blur_n_iterations_spinbox.setMaximum(20)
            self.blur_n_iterations_spinbox.setValue(10)
            self.blur_n_iterations_spinbox.valueChanged.connect(delayed_updater)
            iterations_layout.addWidget(self.blur_n_iterations_spinbox)
            cad_layout.addLayout(iterations_layout)
            self.cad_widget.setLayout(cad_layout)
            blurring_layout.addWidget(self.cad_widget)

            self.gauss_widget = QWidget()
            gauss_layout = QHBoxLayout()
            gauss_layout.setContentsMargins(0, 0, 0, 0)
            gauss_layout.addWidget(QLabel("Sigma", parent=self))
            self.blur_sigma_slider = QDoubleSlider(parent=self)
            self.blur_sigma_slider.setMinimum(0.)
            self.blur_sigma_slider.setMaximum(20.)
            self.blur_sigma_slider.setOrientation(Qt.Horizontal)
            self.blur_sigma_slider.sliderReleased.connect(self.update_blurred_layer)
            gauss_layout.addWidget(self.blur_sigma_slider)
            self.gauss_widget.setLayout(gauss_layout)
            blurring_layout.addWidget(self.gauss_widget)
            self.blurring_type_combobox.currentTextChanged.connect(self.on_blur_selection)

            blur_demo_btn = QPushButton("Try")
            blur_demo_btn.setCheckable(True)
            blur_demo_btn.clicked.connect(self.toggle_blur_demo)
            blurring_layout.addWidget(blur_demo_btn)
            blurring_widget.setLayout(blurring_layout)
            layout.addWidget(blurring_widget, 0, Qt.AlignTop)
            self.on_blur_checked(self.blur_image_checkbox.isChecked())
            self.crop_mask = None

            slice_annotation_widget = CollapsibleWidget("Slice annotation")
            slice_annotation_layout = QVBoxLayout()
            self.slice_segmentation_dropdown = QComboBox()
            self.slice_segmentation_dropdown.addItems(EstimationWorker.ANNOTATION_METHODS)
            slice_annotation_layout.addWidget(self.slice_segmentation_dropdown)
            slice_annotation_widget.setLayout(slice_annotation_layout)
            layout.addWidget(slice_annotation_widget, 0, Qt.AlignTop)

            self.call_button = QPushButton("Run")
            self.call_button.clicked.connect(lambda: self.call_button.setDisabled(True))
            self.call_button.clicked.connect(self.start_estimation)
            layout.addWidget(self.call_button)
            layout.addStretch()
            self.setLayout(layout)
            self._collapsible_group = CollapsibleWidgetGroup([params_widget, blurring_widget, slice_annotation_widget])

        def _add_image(self, layer_name, data, layer_args):
            if 'opacity' not in layer_args:
                layer_args['opacity'] = 0.3
            if 'blending' not in layer_args:
                layer_args['blending'] = "additive"
            if 'rgb' not in layer_args:
                layer_args['rgb'] = False
            self.viewer.add_layer(Image(data, name=layer_name, **layer_args))

        def mask_received(self, data, offset):
            offset = offset.round().astype(int)
            bb_slice = tuple(slice(offs, offs + extent) for offs, extent in zip(offset, data.shape))
            self.labels.layer.data[bb_slice][data > 0] = data[data > 0]
            self.labels.layer.refresh()

        def refresh_layer(self, layer_name):
            if layer_name not in self.viewer.layers:
                return
            layer = self.viewer.layers[layer_name]
            # layer.data = layer.data
            layer.refresh()

        def data_remover(self, name):
            self.viewer.layers.remove(name)

        def start_estimation(self):
            if self.image.layer not in self.viewer.layers:
                self.call_button.setDisabled(False)
                raise ValueError("Missing image layer")
            elif self.points_layer is None:
                self.call_button.setDisabled(False)
                raise ValueError("Missing points layer")
            elif self.labels.layer not in self.viewer.layers:
                self.call_button.setDisabled(False)
                raise ValueError("Missing labels layer")
            image = self.image.layer
            image_data = image.data
            points = self.points_layer
            points_data = points.data.copy()
            z_scale = self.z_scale_spinbox.value()
            alpha = self.alpha_slider.value()
            beta = self.beta_spinner.value()
            use_correction = self.use_correction_checkbox.isChecked()
            estimation_worker = EstimationWorker(self.viewer, self.minimal_contour_widget)
            estimation_worker.initialize(image_data, points_data, use_correction, beta, alpha, self.slice_segmentation_dropdown.currentText(), self.blur_image if self.blur_image_checkbox.isChecked() else None, z_scale)
            estimation_worker.image_data_received.connect(self._add_image)
            estimation_worker.layer_invalidated.connect(self.refresh_layer)
            estimation_worker.mask_data_received.connect(self.mask_received)
            estimation_worker.remove_layer.connect(self.data_remover)
            estimation_worker.all_done.connect(lambda: self.call_button.setDisabled(False))
            estimation_thread = QThread(self)
            estimation_worker.moveToThread(estimation_thread)
            estimation_worker.all_done.connect(estimation_thread.quit)
            estimation_thread.started.connect(estimation_worker.run)
            estimation_thread.finished.connect(estimation_worker.deleteLater)
            estimation_thread.finished.connect(estimation_thread.deleteLater)
            estimation_thread.start()
            # viewer.add_image(data)

        def hook_callbacks(self, estimator, stage, layer_name, layer_params, data_idx=None):

            def initializer(arr, idx):
                self.data_initializer(layer_name, data_idx, layer_params)(arr, idx)

            estimator.hook_stage_data_init_event(stage, initializer)
            estimator.hook_stage_iteration_event(stage, self.data_updater(layer_name))
            estimator.hook_stage_finished_event(stage, self.data_finalizer(layer_name))

        def drag_points_callback(self, layer, event):
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
                bb = pts_2_bb(p1, p2, self.image.layer.data.shape)
                bb_slice = bb_2_slice(bb)
                image = self.image.layer.data[bb_slice]
                crop_mask = generate_sphere_mask(image.shape)
                self.image.layer.visible = False
                cropped_image = self.viewer.add_image(image*crop_mask, colormap=self.image.layer.colormap, translate=bb.min(0), rendering="attenuated_mip")
                self.viewer.layers.selection.active = layer
            yield
            while event.type == "mouse_move":
                yield
            self.viewer.layers.remove(cropped_image)
            self.update_blurred_layer()
            self.image.layer.visible = True

        def on_blur_selection(self, blur_method):
            self.gauss_widget.setVisible(blur_method == "Gaussian")
            self.cad_widget.setVisible(blur_method == "Curvature Anisotropic Diffusion")

        def on_blur_checked(self, is_checked):
            self.blurring_type_widget.setVisible(is_checked)
            self.on_blur_selection(self.blurring_type_combobox.currentText() if is_checked else "")

        def blur_image(self, image):
            blur_type = self.blurring_type_combobox.currentText()
            if blur_type == "Gaussian":
                return gaussian_filter(image, self.blur_sigma_slider.value())
            elif blur_type == "Curvature Anisotropic Diffusion":
                sitk_image = sitk.GetImageFromArray(image.astype(float))
                blurred = sitk.CurvatureAnisotropicDiffusion(
                    sitk_image,
                    conductanceParameter=self.blur_conductance_spinbox.value(),
                    numberOfIterations=self.blur_n_iterations_spinbox.value()
                )
                return sitk.GetArrayFromImage(blurred)
            elif blur_type != "":
                raise NotImplementedError("Unkown blur type %s" % blur_type)

        def toggle_blur_demo(self, is_checked):
            if len(self.points_layer.data) < 2:
                return
            image_layer = self.image.layer
            if image_layer is None:
                return
            if is_checked:
                bb = pts_2_bb(self.points_layer.data[0], self.points_layer.data[1], image_layer.data.shape)
                bb = np.clip(bb, 0, np.asarray(image_layer.data.shape) - 1).round().astype(int)
                bb_slice = bb_2_slice(bb)
                blurred_image = image_layer.data[bb_slice]
                self.crop_mask = generate_sphere_mask(blurred_image.shape)
                offset = bb.min(0)

                blurred = self.blur_image(blurred_image)
                blurred *= self.crop_mask
                self.blurred_layer = self.viewer.add_image(blurred, name="%s cropped" % image_layer.name, translate=np.add(image_layer.translate, offset),
                                   colormap=image_layer.colormap, contrast_limits=image_layer.contrast_limits)
            else:
                if self.blurred_layer:
                    self.viewer.layers.remove(self.blurred_layer)
                    self.blurred_layer = None
                self.crop_mask = None

        def update_blurred_layer(self, *args):
            if self.blurred_layer is None:
                return
            image_layer = self.image.layer
            if image_layer is None:
                return
            bb = pts_2_bb(self.points_layer.data[0], self.points_layer.data[1], image_layer.data.shape)
            bb = np.clip(bb, 0, np.asarray(image_layer.data.shape) - 1).round().astype(int)
            offset = bb.min(0)
            bb_slice = bb_2_slice(bb)
            blurred_image = image_layer.data[bb_slice]
            if self.crop_mask is None or self.crop_mask.shape != blurred_image.shape:
                self.crop_mask = generate_sphere_mask(blurred_image.shape)
            blurred = self.blur_image(blurred_image)
            blurred *= self.crop_mask
            self.blurred_layer.data = blurred
            self.blurred_layer.translate = np.add(image_layer.translate, offset)


    class ShortestPathsWidget(WidgetWithLayerList):
        shapes_data_received = Signal(str, "PyQt_PyObject", "PyQt_PyObject")

        def __init__(self, viewer: napari.Viewer):
            super().__init__(viewer, [("meeting_plane", Image), ("distance_map", Image), ("starting_points", Points)])
            self.shapes_data_received.connect(self.add_shapes)
            self.viewer = viewer
            self.estimator = MinArea.Estimator()
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
            self.update_layer_list(None)

        def add_shapes(self, layer_name, data, layer_args):
            if 'opacity' not in layer_args:
                layer_args['opacity'] = 0.3
            self.viewer.add_layer(Shapes(data, name=layer_name, shape_type="path", **layer_args))

        def plane_2_contour(self):
            if self.distance_map.layer is None or self.meeting_plane.layer is None:
                return
            meeting_plane_layer = self.meeting_plane.layer
            plane_mask = (meeting_plane_layer.data < 0).astype(np.uint8)
            contour = np.squeeze(cv2.findContours(plane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0])
            contour = np.concatenate([np.fliplr(contour), np.ones([len(contour), 1])], axis=1)
            contour = contour @ meeting_plane_layer.rotate.T + meeting_plane_layer.translate
            contour_layer = Points(contour, ndim=self.distance_map.ndim, name="contour points", size=2)
            self.viewer.add_layer(contour_layer)
            for i in range(self.points_layer_dropdown.count()):
                item = self.points_layer_dropdown.itemText(i)
                if item == contour_layer.name:
                    self.points_layer_dropdown.setCurrentIndex(i)
                    break

        def calculate(self, points, distance_map, translate, rotate):
            paths = []
            for point in points:
                shortest_path = self.estimator.resolve_shortest_paths(point, distance_map).copy()
                paths.append(np.fliplr(shortest_path))
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
            points = np.fliplr(points.data - translation).astype(int)
            self.viewer.window.worker2 = Thread(target=self.calculate, args=[points, distance_map, translation, rotation])
            self.viewer.window.worker2.start()
else:
    class MinimalSurfaceWidget(QWidget):
        def __init__(self, *args):
            super().__init__()
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Coming soon...", parent=self))
            layout.addStretch()
            self.setLayout(layout)


def data_event(arr, idx):
    print(arr.shape, idx)

it = [-1]