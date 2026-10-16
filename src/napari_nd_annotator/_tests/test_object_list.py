"""Selecting an object in the object list opens cropped copies of the image and labels layers.

Like the other viewer tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools

import numpy as np
import pytest
from napari.layers.labels._labels_constants import Mode

pytest.importorskip("napari_bbox")

SCALE = (3., 1., 1.)
TRANSLATE = (5., 2., 3.)
BBOX_MIN = np.array([2, 10, 12])
BBOX_MAX = np.array([7, 25, 30])
OBJECT_ID = 4


def _process_events():
    from qtpy.QtWidgets import QApplication
    for _ in range(20):
        QApplication.processEvents()


@pytest.fixture
def viewer():
    import napari
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()
    _process_events()


@pytest.fixture
def scene(viewer):
    """An anisotropic, translated image and labels layer and a bounding box layer with one object."""
    from napari_bbox import BoundingBoxLayer
    from napari_nd_annotator._widgets.object_list import ObjectListWidget
    image = viewer.add_image(np.random.default_rng(0).random((10, 40, 40)), name="image",
                             scale=SCALE, translate=TRANSLATE)
    labels = viewer.add_labels(np.zeros((10, 40, 40), dtype=np.uint16), name="labels",
                               scale=SCALE, translate=TRANSLATE, metadata={"source_image_layer": image})
    # bounding boxes are in world coordinates
    corners = np.where(list(itertools.product((False, True), repeat=3)), BBOX_MAX, BBOX_MIN)
    corners = corners*SCALE + TRANSLATE
    bbox_layer = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(bbox_layer)
    bbox_layer.data = [corners.astype(float)]
    bbox_layer.features = {"label": np.array([OBJECT_ID])}
    object_list = ObjectListWidget(viewer, "Object", bbox_layer, image, labels, None, 1)
    _process_events()
    return image, labels, object_list


def _select_object(viewer, labels, object_list):
    viewer.layers.selection.select_only(labels)
    assert object_list.count() == 1
    object_list.select_item(object_list.item(0))
    _process_events()
    return object_list.crop_image_layer, object_list.crop_mask_layer


def test_bounding_box_is_in_data_coordinates(scene):
    _, _, object_list = scene
    np.testing.assert_array_equal(object_list.item(0).bounding_box, [BBOX_MIN, BBOX_MAX])


def test_crop_labels_source_image_is_the_crop_image(viewer, scene):
    _, labels, object_list = scene
    crop_image, crop_labels = _select_object(viewer, labels, object_list)
    assert crop_labels.metadata["source_image_layer"] is crop_image


def test_crop_layers_line_up_with_the_originals(viewer, scene):
    image, labels, object_list = scene
    crop_image, crop_labels = _select_object(viewer, labels, object_list)
    for original, crop in ((image, crop_image), (labels, crop_labels)):
        np.testing.assert_allclose(crop.scale, original.scale)
        np.testing.assert_array_equal(crop.data.shape, BBOX_MAX - BBOX_MIN + 1)
        # first and last voxel of the crop are at the same world position as the bbox corners in the original
        np.testing.assert_allclose(crop.data_to_world((0, 0, 0)), original.data_to_world(BBOX_MIN))
        np.testing.assert_allclose(crop.data_to_world(np.array(crop.data.shape) - 1),
                                   original.data_to_world(BBOX_MAX))


def test_crop_layers_follow_a_moved_bounding_box(viewer, scene):
    image, labels, object_list = scene
    crop_image, crop_labels = _select_object(viewer, labels, object_list)
    shift = np.array([1, 2, 3])
    corners = np.where(list(itertools.product((False, True), repeat=3)), BBOX_MAX + shift, BBOX_MIN + shift)
    object_list.bounding_box_layer.data = [(corners*SCALE + TRANSLATE).astype(float)]
    object_list.update_items()
    _process_events()
    for original, crop in ((image, crop_image), (labels, crop_labels)):
        np.testing.assert_allclose(crop.data_to_world((0, 0, 0)), original.data_to_world(BBOX_MIN + shift))
        np.testing.assert_array_equal(crop.data, original.data[object_list.item(0).bbox_idx])


@pytest.mark.parametrize("mode", [Mode.PAINT, Mode.ERASE, Mode.FILL, Mode.PAN_ZOOM])
def test_selecting_an_object_keeps_the_labels_mode(viewer, scene, mode):
    _, labels, object_list = scene
    viewer.layers.selection.select_only(labels)
    labels.mode = mode
    _, crop_labels = _select_object(viewer, labels, object_list)
    assert crop_labels.mode == mode


def test_minimal_contour_writes_back_to_the_original_labels(viewer, scene):
    from napari_nd_annotator import AnnotatorWidget
    from napari_nd_annotator._helper_functions import viewer_qt_viewer
    _, labels, object_list = scene
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    crop_image, crop_labels = _select_object(viewer, labels, object_list)

    mc_widget = annotator.ToolsWidget.minimal_contour_widget
    assert mc_widget.labels_layer is crop_labels
    assert mc_widget.image_layer is crop_image
    viewer_qt_viewer(viewer).controls.widgets[crop_labels].mc_button.click()
    _process_events()
    assert crop_labels._overlays["minimal_contour"].enabled

    viewer.dims.set_point(0, crop_labels.data_to_world((3, 0, 0))[0])
    mask = np.zeros(crop_labels.data.shape[1:], dtype=bool)
    mask[2:8, 3:10] = True
    mc_widget._set_mask(mask)
    _process_events()

    assert crop_labels.selected_label == OBJECT_ID
    expected = np.zeros_like(labels.data, dtype=bool)
    expected[BBOX_MIN[0] + 2, BBOX_MIN[1] + 2:BBOX_MIN[1] + 8, BBOX_MIN[2] + 3:BBOX_MIN[2] + 10] = True
    np.testing.assert_array_equal(labels.data == OBJECT_ID, expected)
    assert np.all(labels.data[~expected] == 0)
