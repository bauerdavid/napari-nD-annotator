"""Selecting an object in the object list opens cropped copies of the image and labels layers.

Like the other viewer tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers.labels._labels_constants import Mode

pytest.importorskip("napari_bbox")

from napari_nd_annotator._helper_functions import viewer_qt_viewer  # noqa: E402

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
    assert object_list.count() > 0
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
    # the layer controls show the same tool
    controls = viewer_qt_viewer(viewer).controls.widgets[crop_labels]
    assert controls.button_group.checkedButton().mode == mode


def test_minimal_contour_writes_back_to_the_original_labels(viewer, scene):
    from napari_nd_annotator import AnnotatorWidget
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


def test_minimal_contour_tool_is_carried_over(viewer, scene):
    from napari_nd_annotator import AnnotatorWidget
    _, labels, object_list = scene
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    viewer.layers.selection.select_only(labels)
    viewer_qt_viewer(viewer).controls.widgets[labels].mc_button.click()
    _process_events()
    assert labels._overlays["minimal_contour"].enabled
    _, crop_labels = _select_object(viewer, labels, object_list)
    mc_button = viewer_qt_viewer(viewer).controls.widgets[crop_labels].mc_button
    assert mc_button.isChecked()
    assert crop_labels._overlays["minimal_contour"].enabled


def test_minimal_contour_roi_is_not_used_on_the_cropped_copies(viewer, scene):
    from napari_nd_annotator import AnnotatorWidget
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import         VispyMinimalContourOverlay
    from napari_nd_annotator._widgets.object_list import selected_object
    image, labels, object_list = scene
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    crop_image, crop_labels = _select_object(viewer, labels, object_list)
    selected = selected_object(viewer)
    assert selected[3] is image and selected[3] is not crop_image  # the object belongs to the original image

    mc_widget = annotator.ToolsWidget.minimal_contour_widget
    viewer_qt_viewer(viewer).controls.widgets[crop_labels].mc_button.click()
    _process_events()
    mc_widget._set_image()
    # the contour runs on the small crop itself: it is given as a whole
    assert VispyMinimalContourOverlay._features_shape == crop_image.data.shape[1:]
    assert VispyMinimalContourOverlay._plane_shape is None
    assert VispyMinimalContourOverlay._roi_origin == (0, 0)


def test_new_bounding_boxes_get_consecutive_labels(viewer, scene):
    _, _, object_list = scene
    bbox_layer = object_list.bounding_box_layer
    object_list.indices = 2
    for i, (box_min, box_max) in enumerate((([1, 5, 5], [4, 20, 20]), ([2, 25, 25], [6, 30, 30]))):
        event = SimpleNamespace(type="mouse_press")
        callback = object_list.bounding_box_change(bbox_layer, event)
        next(callback)
        corners = np.where(list(itertools.product((False, True), repeat=3)), box_max, box_min)
        bbox_layer.add(corners*SCALE + TRANSLATE)  # what drawing a box does
        event.type = "mouse_release"
        for _ in callback:
            pass
        _process_events()
        assert list(bbox_layer.features["label"]) == [OBJECT_ID] + list(range(2, 3 + i))
        assert [object_list.item(j).idx for j in range(object_list.count())] == list(bbox_layer.features["label"])


def test_new_bounding_box_labels_continue_after_the_existing_ones(viewer, scene):
    from napari_nd_annotator import ListWidgetBB
    # the widget is opened when the bounding box layer already has objects
    widget = ListWidgetBB(viewer)
    assert widget.list_widget.bounding_box_layer is scene[2].bounding_box_layer
    assert widget.list_widget.next_index() == OBJECT_ID + 1


def _add_objects(object_list, extra_labels):
    """Adds bounding boxes for the given labels next to the first object and updates the list."""
    bbox_layer = object_list.bounding_box_layer
    boxes = [np.asarray(box) for box in bbox_layer.data]
    for i, label in enumerate(extra_labels):
        corners = np.where(list(itertools.product((False, True), repeat=3)), [8, 30 + 3*i, 35], [7, 20 + 3*i, 30])
        boxes.append(corners*SCALE + TRANSLATE)
    bbox_layer.data = boxes
    bbox_layer.features = {"label": np.array([OBJECT_ID] + list(extra_labels))}
    object_list.update_items()
    _process_events()
    assert [object_list.item(i).idx for i in range(object_list.count())] == [OBJECT_ID] + list(extra_labels)


def _delete_box(object_list, index):
    bbox_layer = object_list.bounding_box_layer
    bbox_layer.selected_data = {index}
    bbox_layer.remove_selected()
    _process_events()


def test_deleting_bounding_boxes_updates_the_object_list(viewer, scene):
    _, _, object_list = scene
    _add_objects(object_list, [5, 6])
    _delete_box(object_list, 1)
    assert [object_list.item(i).idx for i in range(object_list.count())] == [OBJECT_ID, 6]
    _delete_box(object_list, 0)
    _delete_box(object_list, 0)
    assert object_list.count() == 0


def test_deleting_the_selected_objects_bounding_box_closes_its_crop(viewer, scene):
    _, labels, object_list = scene
    _add_objects(object_list, [5])
    _select_object(viewer, labels, object_list)
    assert object_list.crop_mask_layer is not None
    _delete_box(object_list, 0)
    assert object_list.crop_image_layer is None and object_list.crop_mask_layer is None
    assert object_list.selected_idx is None
    assert not [layer for layer in viewer.layers if layer.name.startswith("Cropped")]
    assert [object_list.item(i).idx for i in range(object_list.count())] == [5]


def test_deleting_another_bounding_box_keeps_the_crop_selected(viewer, scene):
    _, labels, object_list = scene
    _add_objects(object_list, [5])
    object_list.selected_idx = None
    viewer.layers.selection.select_only(labels)
    object_list.select_item(object_list.item(1))
    _process_events()
    assert object_list.selected_idx == 1
    _delete_box(object_list, 0)
    assert object_list.crop_mask_layer is not None
    assert object_list.selected_idx == 0
