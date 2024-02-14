import warnings

from packaging import version

from .._napari_version import NAPARI_VERSION

try:
    import minimal_surface
except ImportError:
    minimal_surface = None


if minimal_surface is not None:
    from time import sleep
    from copy import deepcopy
    from collections.abc import Iterable
    from napari.utils.notifications import show_info
    from napari.layers._source import layer_source

    from ._utils import WidgetWithLayerList, CollapsibleWidget, CollapsibleWidgetGroup, ProgressWidget, QDoubleSlider, \
    QSymmetricDoubleRangeSlider, ImageProcessingWidget
    from ._utils.delayed_executor import DelayedExecutor
    from ._utils.callbacks import keep_layer_on_top
    from .._helper_functions import layer_dims_displayed
    import napari
    from napari.layers import Image, Labels, Points, Shapes, Layer
    from napari.utils.colormaps.standardize_color import transform_color
    from napari._qt.widgets._slider_compat import QSlider
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
        QSizePolicy,
        QMessageBox
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
        from napari_bbox import BoundingBoxLayer
    except ImportError:
        BoundingBoxLayer = None

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


    it = [-1]
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

        def __init__(self, viewer, minimal_contour_widget, minimal_surface_widget, parent=None):
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
            self.init_slice_annotation_fun()
        
        def finish_slice_annotation(self):
            self.finish_slice_annotation_fun()
            self.slice_annotation = self.slice_labels_layer.data > 0
            self.viewer.layers.remove(self.slice_image_layer)
            self.viewer.layers.remove(self.slice_labels_layer)
            self.viewer.dims.ndisplay = self.prev_ndisplay
            self.viewer.camera.center = self.prev_camera_center
            self.viewer.camera.zoom = self.prev_camera_zoom
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
            self.minimal_contour_widget.image.layer = self.slice_image_layer
            self.minimal_contour_widget.labels.layer = self.slice_labels_layer
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

            estimator.hook_transform_calculated_event(tform_calculated)

            def center_calculated(center):
                self.center_mat = np.flip(center.copy())

            estimator.hook_plane_center_calculated_event(center_calculated)

        def run(self):
            import time
            self.minimal_surface_widget.call_button.clicked.disconnect(self.minimal_surface_widget.start_estimation)
            def stop():
                self.stop_requested = True
                self.minimal_surface_widget.call_button.setText("Stopping...")
                self.finish_slice_annotation()
            self.minimal_surface_widget.call_button.clicked.connect(stop)
            self.minimal_surface_widget.call_button.setText("Stop")

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
                self.minimal_surface_widget.call_button.clicked.connect(self.minimal_surface_widget.start_estimation)
                self.minimal_surface_widget.call_button.setText("Run")

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
            mask = self.slice_annotation.astype(float) if self.slice_annotation is not None else None
            self.slice_annotation = None
            return mask


    class ClippingWidget(QWidget):
        def __init__(self, viewer: napari.Viewer, layers=None):
            super().__init__()
            self._layers = []
            self._viewer = viewer
            self._ghost_layers = dict()
            layout = QVBoxLayout()
            self.sliders = []
            self.slider_checkboxes = []
            self._clipping_planes = []
            self.n_checked = 0
            for i in range(3):
                slider_layout = QHBoxLayout()
                slider_checkbox = QCheckBox("%s" % i)
                slider_checkbox.clicked.connect(self.on_checked_handler(i))
                slider_layout.addWidget(slider_checkbox)
                slider = QSymmetricDoubleRangeSlider(Qt.Horizontal)
                slider.valueChanged.connect(self.range_change_handler(i))
                slider.setEnabled(slider_checkbox.isChecked())
                slider_layout.addWidget(slider)
                layout.addLayout(slider_layout)
                self.sliders.append(slider)
                self.slider_checkboxes.append(slider_checkbox)
            self.setLayout(layout)
            self.initialize_clipping_planes()
            self.layers = layers
            self._viewer.dims.events.ndisplay.connect(self.on_ndisplay_changed)

        @property
        def layers(self):
            return list(self._layers)

        @layers.setter
        def layers(self, new_layers):
            new_layers = list(new_layers) if isinstance(new_layers, Iterable)\
                else [new_layers] if new_layers else []
            if self._layers:
                for layer in self.layers:
                    if layer not in new_layers:
                        self.remove_layer(layer)
                    else:
                        new_layers.remove(layer)
            for layer in new_layers:
                self.add_layer(layer)
            self.update_ranges()

        def add_layer(self, layer):
            if layer is None:
                return
            if layer in self.layers:
                return
            self._layers.append(layer)
            self._ghost_layers[layer] = None
            self.update_ghost_layers()
            layer.events.scale.connect(self.update_ranges)
            layer.experimental_clipping_planes = self._clipping_planes
            layer.events.data.connect(self.on_layer_data_change)
            self.update_ranges()

        def remove_layer(self, layer):
            if layer is None:
                return
            if layer not in self.layers:
                return
            self._layers.remove(layer)
            if self._ghost_layers[layer] is not None:
                self._viewer.layers.remove(self._ghost_layers[layer])
            del self._ghost_layers[layer]
            layer.events.scale.disconnect(self.update_ranges)
            layer.events.data.disconnect(self.on_layer_data_change)
            layer.experimental_clipping_planes = []
            self.update_ranges()

        def on_layer_data_change(self, event):
            layer = event.source
            if layer not in self._ghost_layers:
                print("%s not in ghost layers!" % layer.name if layer else None)
                return
            ghost_layer = self._ghost_layers[event.source]
            if ghost_layer is None:
                return
            ghost_layer.data = event.source.data

        def range_change_handler(self, idx):
            def on_range_change(value):
                self._clipping_planes[idx * 2]["position"][idx] = value[0]
                self._clipping_planes[idx*2+1]["position"][idx] = value[1]
                self.update_clipping_planes()
            return on_range_change

        def on_checked_handler(self, idx):
            def on_checked(value):
                self.n_checked += 1 if value else -1
                self.update_ghost_layers()
                if value != self.sliders[idx].isEnabled():
                    self.sliders[idx].setEnabled(value)
                self.update_clipping_planes()
            return on_checked

        def update_ghost_layers(self):
            for layer in self._layers:
                ghost = self._ghost_layers[layer]
                if ghost is None and self.n_checked>0:
                    data, state, type_str = layer.as_layer_data_tuple()
                    state["name"] = u"\U0001F47B " + layer.name
                    state["opacity"] /= 2.
                    state["experimental_clipping_planes"] = []
                    if NAPARI_VERSION <= version.parse("0.4.16") and "interpolation" in state:
                        interpolation = state["interpolation"]
                        state["interpolation"] = "nearest"
                    with layer_source(parent=layer):
                        new = Layer.create(deepcopy(data), state, type_str)
                    self._viewer.add_layer(new)
                    if NAPARI_VERSION <= version.parse("0.4.16"):
                        new.interpolation = interpolation
                    self._viewer.layers.link_layers([layer, new])
                    self._viewer.layers.unlink_layers([layer, new], ["opacity", "experimental_clipping_planes", "blending"])
                    self._viewer.layers.move(self._viewer.layers.index(new), self._viewer.layers.index(layer)+1)
                    self._ghost_layers[layer] = new
                elif ghost is not None and self.n_checked == 0:
                    self._viewer.layers.remove(ghost)
                    self._ghost_layers[layer] = None

        def remove_ghost_layers(self):
            for layer in self._layers:
                ghost = self._ghost_layers[layer]
                if ghost is not None:
                    self._viewer.layers.remove(ghost)
                    self._ghost_layers[layer] = None

        def on_ndisplay_changed(self, e):
            if e.value == 2:
                self.remove_ghost_layers()
            else:
                self.update_ghost_layers()

        def update_ranges(self):
            if not self.layers:
                for slider in self.sliders:
                    slider.setEnabled(False)
                return
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
                slider.setEnabled(not np.any(np.equal(extent[:, s_idx], np.asarray([nan, negnan]))))
                slider.setMinimum(extent[0, s_idx])
                slider.setMaximum(extent[1, s_idx])
            self.update_clipping_planes()

        def initialize_clipping_planes(self):
            new_clipping_planes = []
            for s_idx, slider in enumerate(self.sliders):
                start, end = slider.value()
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

        def update_clipping_planes(self):
            new_clipping_planes = []
            for s_idx, checkbox in enumerate(self.slider_checkboxes):
                if not checkbox.isChecked():
                    continue
                new_clipping_planes.extend(self._clipping_planes[s_idx*2:s_idx*2+2])
            for layer in self.layers:
                layer.experimental_clipping_planes = new_clipping_planes


    class ImageSliceWidget(QWidget):
        def __init__(self, viewer, layer=None):
            super().__init__()
            self.viewer = viewer
            self._layer = None
            self.slice_layer = None
            self._ctrl_down = False
            self._mouse_drag_callbacks = []
            layout = QVBoxLayout()
            dim_layout = QHBoxLayout()
            dim_layout.addWidget(QLabel("Axis"))
            self.dim_spinbox = QSpinBox(self)
            self.dim_spinbox.setMaximum(0)
            self.dim_spinbox.valueChanged.connect(lambda _: self.update_data())
            dim_layout.addWidget(self.dim_spinbox)
            layout.addLayout(dim_layout)
            pos_layout = QHBoxLayout()
            pos_layout.addWidget(QLabel("Position"))
            self.position_slider = QSlider(Qt.Horizontal)
            self.position_slider.setMaximum(0)
            self.position_slider.valueChanged.connect(lambda: self.update_data(False))
            pos_layout.addWidget(self.position_slider)
            layout.addLayout(pos_layout)
            self.clipping_widget = ClippingWidget(self.viewer, self.layer)
            layout.addWidget(self.clipping_widget)
            self.update_widgets()
            self.update_data()
            self.setLayout(layout)
            self.viewer.dims.events.ndisplay.connect(self.on_ndisplay_changed)
            self.layer = layer

        @property
        def position(self):
            return self.position_slider.value()

        @property
        def dimension(self):
            return self.dim_spinbox.value()

        @property
        def layer(self):
            return self._layer

        @layer.setter
        def layer(self, new_layer):
            self.clipping_widget.remove_layer(self._layer)
            self._layer = new_layer
            self.clipping_widget.add_layer(self._layer)
            self.update_widgets()
            if self.viewer.dims.ndisplay == 3:
                self.create_slice_layer()

        def scroll_slice(self, layer, event):
            if "Alt" not in event.modifiers:
                return
            self.position_slider.setValue(int(self.position-event.delta[0]))

        def set_widget_enabled(self):
            is_enabled = self.layer is not None and self.viewer.dims.ndisplay == 3
            self.dim_spinbox.setEnabled(is_enabled)
            self.position_slider.setEnabled(is_enabled)

        def remove_slice_layer(self):
            if self.slice_layer is not None:
                self.viewer.layers.remove(self.slice_layer)
                self.clipping_widget.remove_layer(self.slice_layer)
                self.slice_layer = None

        def create_slice_layer(self):
            self.remove_slice_layer()
            if self.layer is None:
                return
            self.slice_layer = self.viewer.add_image(data=np.empty((1, 1, 1)), colormap="red", scale=self.layer.scale, blending="additive", experimental_clipping_planes=self.layer.experimental_clipping_planes)
            self.clipping_widget.add_layer(self.slice_layer)
            self.slice_layer.mouse_drag_callbacks.extend(self._mouse_drag_callbacks)
            self.slice_layer.mouse_wheel_callbacks.append(self.scroll_slice)
            self.update_data()
            self.slice_layer.reset_contrast_limits()

        def update_widgets(self):
            if self.layer is not None:
                self.dim_spinbox.setMaximum(self.layer.data.ndim-1)
                self.position_slider.setMaximum(self.layer.data.shape[self.dimension]-1)
            self.set_widget_enabled()

        def update_data(self, update_widgets=True):
            if self.slice_layer is None:
                return
            if update_widgets:
                self.update_widgets()
            dim = self.dimension
            idx = tuple(slice(self.position, self.position + 1) if i == dim else slice(None)
                        for i in range(self.layer.data.ndim))
            self.slice_layer.data = self.layer.data[idx]
            self.slice_layer.translate = [self.position*self.layer.scale[i] if i == dim else 0 for i in range(self.layer.data.ndim)]

        def on_ndisplay_changed(self):
            if self.viewer.dims.ndisplay == 2:
                self.remove_slice_layer()
            elif self.viewer.dims.ndisplay == 3:
                self.create_slice_layer()
            self.set_widget_enabled()

        def add_mouse_drag_callback(self, callback):
            self._mouse_drag_callbacks.append(callback)
            if self.slice_layer is not None:
                self.slice_layer.mouse_drag_callbacks.append(callback)


    class MinimalSurfaceWidget(WidgetWithLayerList):
        def __init__(self, viewer: napari.Viewer, minimal_contour_widget, parent=None):
            super().__init__(viewer, [("image", Image), ("labels", Labels)], "nd_annotator_ms", parent=parent)
            self.labels.combobox.currentIndexChanged.connect(self.on_labels_changed)
            self._prev_labels = None
            self.blurred_layer = None
            self.blurring_progress_widget = ProgressWidget(message="blurring...")
            self.estimation_progress_widget = ProgressWidget(message="Annotating...")
            self.minimal_contour_widget = minimal_contour_widget
            self.points_layer = self.viewer.add_points(ndim=3, size=2, name="3D Anchors [DO NOT REMOVE]")
            self.points_layer.events.connect(ColorPairsCallback())
            self.points_layer.mouse_drag_callbacks.append(self.drag_points_callback)
            self.viewer.layers.events.inserted.connect(keep_layer_on_top(self.points_layer))
            self.viewer.layers.events.moved.connect(keep_layer_on_top(self.points_layer))
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignTop)
            # Parameters for the estimator
            params_widget = CollapsibleWidget("Image features")
            params_widget.setContentsMargins(0, 0, 0, 0)
            params_layout = QVBoxLayout()
            self.image_feature_combobox = QComboBox(self)
            self.image_feature_combobox.addItem("Gradient")
            self.image_feature_combobox.addItem("Custom")
            params_layout.addWidget(self.image_feature_combobox)
            self.add_stored_widget("image_feature_combobox")
            self.alpha_beta_widget = QWidget()
            self.alpha_beta_widget.setContentsMargins(0, 0, 0, 0)
            alpha_beta_layout = QVBoxLayout()
            alpha_layout = QHBoxLayout()
            alpha_layout.addWidget(QLabel("\u03B1", parent=self))
            self.alpha_slider = QDoubleSlider(parent=self)
            self.alpha_slider.setMinimum(0.)
            self.alpha_slider.setMaximum(1.)
            self.alpha_slider.setDecimals(3)
            self.alpha_slider.setOrientation(Qt.Horizontal)
            alpha_layout.addWidget(self.alpha_slider)
            self.add_stored_widget("alpha_slider")
            alpha_beta_layout.addLayout(alpha_layout)

            beta_layout = QHBoxLayout()
            beta_layout.addWidget(QLabel("\u03B2", parent=self))
            self.beta_spinner = QDoubleSpinBox()
            self.beta_spinner.setMinimum(.01)
            self.beta_spinner.setMaximum(500.)
            self.beta_spinner.setSingleStep(1.)
            self.beta_spinner.setSizePolicy(QSizePolicy.Policy.Expanding, self.beta_spinner.sizePolicy().verticalPolicy())
            beta_layout.addWidget(self.beta_spinner)
            self.add_stored_widget("beta_spinner")
            alpha_beta_layout.addLayout(beta_layout)

            self.alpha_beta_widget.setLayout(alpha_beta_layout)
            params_layout.addWidget(self.alpha_beta_widget)
            self.custom_feature_widget = ImageProcessingWidget(self.image.layer, self.viewer, "min_surf_script", self)
            params_layout.addWidget(self.custom_feature_widget)
            self.image_feature_combobox.currentTextChanged.connect(lambda t: self.custom_feature_widget.setVisible(t == "Custom"))
            self.custom_feature_widget.setVisible(self.image_feature_combobox.currentText() == "Custom")
            iterations_layout = QHBoxLayout()
            iterations_layout.addWidget(QLabel("iterations", parent=self))
            self.iterations_slider = QSlider(parent=self)
            self.iterations_slider.setMinimum(1)
            self.iterations_slider.setMaximum(10000)
            self.iterations_slider.setOrientation(Qt.Horizontal)
            iterations_layout.addWidget(self.iterations_slider)
            self.add_stored_widget("iterations_slider")
            params_layout.addLayout(iterations_layout)

            z_scale_layout = QHBoxLayout()
            z_scale_layout.addWidget(QLabel("Z-scale"))
            self.z_scale_spinbox = QDoubleSpinBox()
            self.z_scale_spinbox.setMinimum(0.01)
            self.z_scale_spinbox.setMaximum(20)
            self.z_scale_spinbox.setSingleStep(1.)
            # self.z_scale_spinbox.valueChanged.connect(self) TODO
            z_scale_layout.addWidget(self.z_scale_spinbox)
            self.add_stored_widget("z_scale_spinbox")
            params_layout.addLayout(z_scale_layout)

            self.use_correction_checkbox = QCheckBox("Use correction")
            params_layout.addWidget(self.use_correction_checkbox)
            self.add_stored_widget("use_correction_checkbox")
            params_widget.setLayout(params_layout)
            layout.addWidget(params_widget, 0, Qt.AlignTop)

            # Blurring parameters
            blurring_widget = CollapsibleWidget("Blurring", self)
            blurring_layout = QVBoxLayout()
            self.blur_image_checkbox = QCheckBox("Blur image")
            self.blur_image_checkbox.clicked.connect(self.on_blur_checked)
            blurring_layout.addWidget(self.blur_image_checkbox)
            self.add_stored_widget("blur_image_checkbox")
            self.blurring_type_widget = QWidget()
            blurring_type_layout = QVBoxLayout()
            blurring_type_layout.setContentsMargins(0, 0, 0, 0)
            blurring_type_layout.addWidget(QLabel("Blurring type", self))
            self.blurring_type_combobox = QComboBox()
            self.blurring_type_combobox.addItems(["Gaussian", "Curvature Anisotropic Diffusion"])
            self.blurring_type_combobox.setSizePolicy(QSizePolicy.Policy.Ignored, self.blurring_type_combobox.sizePolicy().verticalPolicy())
            self.blurring_type_combobox.currentTextChanged.connect(self.update_blurred_layer)
            blurring_type_layout.addWidget(self.blurring_type_combobox)
            self.add_stored_widget("blurring_type_combobox")
            self.blurring_type_widget.setLayout(blurring_type_layout)
            blurring_layout.addWidget(self.blurring_type_widget)
            self.cad_widget = QWidget()
            cad_layout = QVBoxLayout()
            cad_layout.setContentsMargins(0, 0, 0, 0)
            conductance_layout = QHBoxLayout()
            conductance_layout.addWidget(QLabel("Conductance"))
            delayed_updater = DelayedExecutor(self.update_blurred_layer)
            delayed_updater.processing.connect(lambda _: self.blurring_progress_widget.setVisible(blur_demo_btn.isChecked()))
            delayed_updater.processed.connect(lambda _: self.blurring_progress_widget.setVisible(False))
            self.blur_conductance_spinbox = QDoubleSpinBox()
            self.blur_conductance_spinbox.setMinimum(0)
            self.blur_conductance_spinbox.setMaximum(100)
            self.blur_conductance_spinbox.valueChanged.connect(delayed_updater)
            conductance_layout.addWidget(self.blur_conductance_spinbox)
            self.add_stored_widget("blur_conductance_spinbox")
            cad_layout.addLayout(conductance_layout)

            iterations_layout = QHBoxLayout()
            iterations_layout.addWidget(QLabel("# iterations"))
            self.blur_n_iterations_spinbox = QSpinBox()
            self.blur_n_iterations_spinbox.setMinimum(1)
            self.blur_n_iterations_spinbox.setMaximum(20)
            self.blur_n_iterations_spinbox.valueChanged.connect(delayed_updater)
            iterations_layout.addWidget(self.blur_n_iterations_spinbox)
            self.add_stored_widget("blur_n_iterations_spinbox")
            cad_layout.addLayout(iterations_layout)
            self.cad_widget.setLayout(cad_layout)
            blurring_layout.addWidget(self.cad_widget)

            self.gauss_widget = QWidget(self)
            gauss_layout = QHBoxLayout()
            gauss_layout.setContentsMargins(0, 0, 0, 0)
            gauss_layout.addWidget(QLabel("Sigma", parent=self))
            self.blur_sigma_slider = QDoubleSlider(parent=self)
            self.blur_sigma_slider.setMinimum(0.)
            self.blur_sigma_slider.setMaximum(20.)
            self.blur_sigma_slider.setOrientation(Qt.Horizontal)
            self.blur_sigma_slider.sliderReleased.connect(self.update_blurred_layer)
            gauss_layout.addWidget(self.blur_sigma_slider)
            self.add_stored_widget("blur_sigma_slider")
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
            self.add_stored_widget("slice_segmentation_dropdown")
            self.use_meeting_points_checkbox = QCheckBox("Slice from meeting points")
            self.use_meeting_points_checkbox.setChecked(True)
            slice_annotation_layout.addWidget(self.use_meeting_points_checkbox)
            slice_annotation_widget.setLayout(slice_annotation_layout)
            layout.addWidget(slice_annotation_widget, 0, Qt.AlignTop)

            self.call_button = QPushButton("Run")
            self.call_button.clicked.connect(self.start_estimation)
            layout.addWidget(self.call_button)
            self.slice_widget = ImageSliceWidget(self.viewer, self.image.layer)
            # self.slice_widget.clipping_widget.add_layer(self.points_layer)
            self.slice_widget.add_mouse_drag_callback(self.on_slice_clicked)
            self.image.combobox.currentIndexChanged.connect(self.set_slice_widget_image)
            self.image.combobox.currentIndexChanged.connect(self.update_custom_feature_image)
            layout.addWidget(self.slice_widget)
            layout.addStretch()
            self.setLayout(layout)
            self._collapsible_group = CollapsibleWidgetGroup([params_widget, blurring_widget, slice_annotation_widget])
            self.on_labels_changed()

        def on_labels_changed(self, idx=None):
            self.slice_widget.clipping_widget.remove_layer(self._prev_labels)
            self.slice_widget.clipping_widget.add_layer(self.labels.layer)
            self._prev_labels = self.labels.layer

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
            self.labels.layer.data[bb_slice][data > 0] = self.labels.layer.data.max()+1
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
                warnings.warn("Missing image layer")
                return
            if len(self.points_layer.data) == 0:
                warnings.warn("No points were added")
                return
            if len(self.points_layer.data) == 1:
                warnings.warn("Not enough points")
                return
            if self.labels.layer is None:
                answer = QMessageBox.question(
                    self,
                    "Missing Labels layer",
                    "Create Labels layer?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if answer == QMessageBox.StandardButton.Yes:
                    new_labels = self.viewer.add_labels(
                        np.zeros(self.image.layer.data.shape[:self.image.layer.ndim], dtype=np.uint16))
                    self.labels.layer = new_labels
                else:
                    return
            elif not np.array_equal(
                    self.labels.layer.data.shape[:self.labels.layer.ndim],
                    self.image.layer.data.shape[:self.image.layer.ndim]
            ):
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
                    return
            image = self.image.layer
            image_data = image.data
            phi_func = None if self.image_feature_combobox.currentText() == "Gradient" else self.custom_feature_widget.execute_script
            points = self.points_layer
            points_data = points.data.copy()
            z_scale = self.z_scale_spinbox.value()
            alpha = self.alpha_slider.value()
            beta = self.beta_spinner.value()
            n_iter = self.iterations_slider.value()
            use_correction = self.use_correction_checkbox.isChecked()
            use_gradient = self.image_feature_combobox.currentText() == "Gradient"
            use_meeting_plane_points = self.use_meeting_points_checkbox.isChecked()
            estimation_worker = EstimationWorker(self.viewer, self.minimal_contour_widget, self)
            estimation_worker.initialize(image_data, phi_func, points_data, use_correction, beta, alpha, n_iter, use_meeting_plane_points, self.slice_segmentation_dropdown.currentText(), self.blur_image if self.blur_image_checkbox.isChecked() else None, z_scale, use_gradient)
            estimation_worker.image_data_received.connect(self._add_image)
            estimation_worker.layer_invalidated.connect(self.refresh_layer)
            estimation_worker.mask_data_received.connect(self.mask_received)
            estimation_worker.remove_layer.connect(self.data_remover)
            estimation_worker.slice_annotations_done.connect(self.show_estimation_progress)
            estimation_worker.object_annotated.connect(lambda i: self.estimation_progress_widget.setValue(i+1))
            estimation_thread = QThread(self)
            estimation_worker.moveToThread(estimation_thread)
            estimation_worker.all_done.connect(estimation_thread.quit)
            estimation_worker.all_done.connect(lambda: self.estimation_progress_widget.setVisible(False))
            estimation_thread.started.connect(estimation_worker.run)
            estimation_thread.finished.connect(estimation_worker.deleteLater)
            estimation_thread.finished.connect(estimation_thread.deleteLater)
            if BoundingBoxLayer is not None:
                estimation_worker.bb_layer = BoundingBoxLayer(
                    [[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]], scale=(z_scale, 1., 1.))
                self.viewer.add_layer(estimation_worker.bb_layer)
                estimation_thread.finished.connect(lambda: self.viewer.layers.remove(estimation_worker.bb_layer))
            estimation_thread.start()
            # viewer.add_image(data)

        def show_estimation_progress(self, n_objects):
            self.estimation_progress_widget.setMaximum(n_objects)
            self.estimation_progress_widget.setValue(0)
            self.estimation_progress_widget.setVisible(True)

        # def hook_callbacks(self, estimator, stage, layer_name, layer_params, data_idx=None):
        #
        #     def initializer(arr, idx):
        #         self.data_initializer(layer_name, data_idx, layer_params)(arr, idx)
        #
        #     estimator.hook_stage_data_init_event(stage, initializer)
        #     estimator.hook_stage_iteration_event(stage, self.data_updater(layer_name))
        #     estimator.hook_stage_finished_event(stage, self.data_finalizer(layer_name))

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
                cropped_image = self.viewer.add_image(
                    image*crop_mask,
                    colormap=self.image.layer.colormap,
                    translate=bb.min(0)*self.image.layer.scale,
                    scale=self.image.layer.scale,
                    rendering="attenuated_mip"
                )
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
                bb = pts_2_bb(self.points_layer.data[0], self.points_layer.data[1], image_layer.data.shape, image_layer.scale)
                bb = bb.round().astype(int)
                bb_slice = bb_2_slice(bb)
                blurred_image = image_layer.data[bb_slice]
                self.crop_mask = generate_sphere_mask(blurred_image.shape)
                offset = bb.min(0)

                blurred = self.blur_image(blurred_image)
                blurred *= self.crop_mask
                self.blurred_layer = self.viewer.add_image(
                    blurred,
                    name="%s cropped" % image_layer.name,
                    translate=np.add(image_layer.translate, offset),
                    scale=image_layer.scale,
                    colormap=image_layer.colormap,
                    contrast_limits=image_layer.contrast_limits
                )
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
            bb = pts_2_bb(self.points_layer.data[0], self.points_layer.data[1], image_layer.data.shape, image_layer.scale)
            bb = bb.round().astype(int)
            offset = bb.min(0)
            bb_slice = bb_2_slice(bb)
            blurred_image = image_layer.data[bb_slice]
            if self.crop_mask is None or self.crop_mask.shape != blurred_image.shape:
                self.crop_mask = generate_sphere_mask(blurred_image.shape)
            blurred = self.blur_image(blurred_image)
            blurred *= self.crop_mask
            self.blurred_layer.data = blurred
            self.blurred_layer.translate = np.add(image_layer.translate, offset)

        def set_slice_widget_image(self, _):
            self.slice_widget.layer = self.image.layer

        def update_custom_feature_image(self, _):
            if self.image.layer is not None:
                self.custom_feature_widget.image = self.image.layer.data

        def on_slice_clicked(self, layer, event):
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
                    for i, slider in enumerate(self.slice_widget.clipping_widget.sliders):
                        range = slider.value()
                        range_center = (range[0] + range[1])/2
                        new_center = int(point[i])
                        offset = new_center-range_center
                        slider.setValue((range[0]+offset, range[1]+offset))
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
else:
    MinimalSurfaceWidget = None

