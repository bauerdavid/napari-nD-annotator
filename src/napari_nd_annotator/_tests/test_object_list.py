"""Selecting an object in the object list moves the view to it; the original layers are used, no copies are made.

Like the other viewer tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers.labels._labels_constants import Mode

pytest.importorskip("napari_bbox")

from napari_nd_annotator._helper_functions import (  # noqa: E402
    layer_slice_indices, viewer_camera, viewer_canvas_size, viewer_qt_viewer)
from napari_nd_annotator._widgets.object_list import selected_object, selected_object_changed  # noqa: E402

SCALE = (3., 1., 1.)
TRANSLATE = (5., 2., 3.)
BBOX_MIN = np.array([2, 10, 12])
BBOX_MAX = np.array([7, 25, 30])
OBJECT_ID = 4


def _process_events():
    from qtpy.QtWidgets import QApplication
    for _ in range(20):
        QApplication.processEvents()


def _wait(ms=120):
    """Lets the throttled updates (e.g. of the icons) happen."""
    from qtpy.QtTest import QTest
    QTest.qWait(ms)
    _process_events()


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


def _select_object(viewer, labels, object_list, index=0):
    viewer.layers.selection.select_only(labels)
    assert object_list.count() > index
    object_list.select_item(object_list.item(index))
    _process_events()


def _view(viewer):
    camera = viewer_camera(viewer)
    return tuple(viewer.dims.point), tuple(camera.center), camera.zoom


def _move_view_away(viewer, labels):
    """Looks at another slice and part of the image than the object."""
    viewer.dims.set_point(0, labels.data_to_world((8, 0, 0))[0])
    camera = viewer_camera(viewer)
    camera.center = (0, 30., 8.)
    camera.zoom = 3.
    _process_events()
    return _view(viewer)


def _assert_view(viewer, expected):
    point, center, zoom = _view(viewer)
    np.testing.assert_allclose(point, expected[0])
    np.testing.assert_allclose(center, expected[1])
    assert zoom == pytest.approx(expected[2])


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


def _move_bounding_box(object_list, shift):
    corners = np.where(list(itertools.product((False, True), repeat=3)), BBOX_MAX + shift, BBOX_MIN + shift)
    object_list.bounding_box_layer.data = [(corners*SCALE + TRANSLATE).astype(float)]
    object_list.update_items()
    _process_events()


def _annotator(viewer):
    from napari_nd_annotator import AnnotatorWidget
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    return annotator


def test_bounding_box_is_in_data_coordinates(scene):
    _, _, object_list = scene
    np.testing.assert_array_equal(object_list.item(0).bounding_box, [BBOX_MIN, BBOX_MAX])


def test_selecting_an_object_creates_no_layers_and_copies_no_data(viewer, scene):
    image, labels, object_list = scene
    layers, image_data, labels_data = list(viewer.layers), image.data, labels.data
    _select_object(viewer, labels, object_list)
    assert list(viewer.layers) == layers
    assert image.data is image_data and labels.data is labels_data
    assert np.shares_memory(image.data, image_data) and np.shares_memory(labels.data, labels_data)
    assert list(viewer.layers.selection) == [labels]
    assert image.visible and labels.visible
    assert labels.selected_label == OBJECT_ID
    assert object_list.clipper is None
    for layer in (image, labels):
        assert len(layer.experimental_clipping_planes) == 0
    # the annotation stays on the original layers
    assert selected_object(viewer)[3] is image
    assert "source_image_layer" in labels.metadata and labels.metadata["source_image_layer"] is image


def test_selection_gives_labels_without_a_source_image_one(viewer, scene):
    image, labels, object_list = scene
    del labels.metadata["source_image_layer"]
    _select_object(viewer, labels, object_list)
    assert labels.metadata["source_image_layer"] is image


def test_selection_moves_the_slice_and_the_camera_to_the_object(viewer, scene):
    image, labels, object_list = scene
    _move_view_away(viewer, labels)
    _select_object(viewer, labels, object_list)
    # the middle slice of the object
    assert 4 <= labels.world_to_data(viewer.dims.point)[0] <= 5
    center = np.asarray(image.data_to_world((BBOX_MIN + BBOX_MAX) / 2))
    camera = viewer_camera(viewer)
    np.testing.assert_allclose(camera.center[-2:], center[1:])
    height, width = viewer_canvas_size(viewer)
    if height > 0 and width > 0:
        extent = (BBOX_MAX - BBOX_MIN + 1)[1:] * np.asarray(SCALE)[1:]
        assert camera.zoom == pytest.approx(0.9 * min(height / extent[0], width / extent[1]))


def test_selection_with_a_transposed_view(viewer, scene):
    image, labels, object_list = scene
    viewer.dims.order = (1, 0, 2)  # rows show axis 0, the slices run along axis 1
    _select_object(viewer, labels, object_list)
    center = np.asarray(image.data_to_world((BBOX_MIN + BBOX_MAX) / 2))
    np.testing.assert_allclose(viewer_camera(viewer).center[-2:], center[[0, 2]])
    assert 17 <= labels.world_to_data(viewer.dims.point)[1] <= 18


def test_unselecting_restores_the_view(viewer, scene):
    _, labels, object_list = scene
    before = _move_view_away(viewer, labels)
    _select_object(viewer, labels, object_list)
    assert _view(viewer) != before
    object_list.select_item(object_list.item(0))  # clicking the selected object again unselects it
    _process_events()
    assert object_list.selected_idx is None
    _assert_view(viewer, before)


def test_selecting_another_object_keeps_the_first_saved_view(viewer, scene):
    image, labels, object_list = scene
    _add_objects(object_list, [5])
    before = _move_view_away(viewer, labels)
    _select_object(viewer, labels, object_list)
    first_object_view = _view(viewer)
    _select_object(viewer, labels, object_list, 1)
    assert object_list.selected_idx == 1 and labels.selected_label == 5
    second = object_list.item(1).bounding_box
    center = np.asarray(image.data_to_world(second.mean(axis=0)))
    np.testing.assert_allclose(viewer_camera(viewer).center[-2:], center[1:])
    assert _view(viewer) != first_object_view
    object_list.select_item(object_list.item(1))
    _process_events()
    _assert_view(viewer, before)


@pytest.mark.parametrize("mode", [Mode.PAINT, Mode.ERASE, Mode.FILL, Mode.PAN_ZOOM])
def test_selecting_an_object_keeps_the_labels_mode(viewer, scene, mode):
    _, labels, object_list = scene
    viewer.layers.selection.select_only(labels)
    labels.mode = mode
    _select_object(viewer, labels, object_list)
    assert labels.mode == mode
    # the layer controls show the same tool
    controls = viewer_qt_viewer(viewer).controls.widgets[labels]
    assert controls.button_group.checkedButton().mode == mode


def test_minimal_contour_writes_to_the_original_labels(viewer, scene):
    image, labels, object_list = scene
    annotator = _annotator(viewer)
    _select_object(viewer, labels, object_list)

    mc_widget = annotator.ToolsWidget.minimal_contour_widget
    assert mc_widget.labels_layer is labels
    assert mc_widget.image_layer is image
    viewer_qt_viewer(viewer).controls.widgets[labels].mc_button.click()
    _process_events()
    assert labels._overlays["minimal_contour"].enabled

    viewer.dims.set_point(0, labels.data_to_world((4, 0, 0))[0])
    slice_index = layer_slice_indices(labels)[0]
    assert BBOX_MIN[0] <= slice_index <= BBOX_MAX[0]
    mask = np.zeros(labels.data.shape[1:], dtype=bool)
    mask[BBOX_MIN[1] + 2:BBOX_MIN[1] + 8, BBOX_MIN[2] + 3:BBOX_MIN[2] + 10] = True
    labels_data = labels.data
    mc_widget._set_mask(mask)
    _process_events()

    assert labels.data is labels_data
    assert labels.selected_label == OBJECT_ID
    expected = np.zeros_like(labels.data, dtype=bool)
    expected[slice_index, BBOX_MIN[1] + 2:BBOX_MIN[1] + 8, BBOX_MIN[2] + 3:BBOX_MIN[2] + 10] = True
    np.testing.assert_array_equal(labels.data == OBJECT_ID, expected)
    assert np.all(labels.data[~expected] == 0)


def test_minimal_contour_tool_is_carried_over(viewer, scene):
    _, labels, object_list = scene
    _annotator(viewer)
    viewer.layers.selection.select_only(labels)
    viewer_qt_viewer(viewer).controls.widgets[labels].mc_button.click()
    _process_events()
    assert labels._overlays["minimal_contour"].enabled
    _select_object(viewer, labels, object_list)
    assert viewer_qt_viewer(viewer).controls.widgets[labels].mc_button.isChecked()
    assert labels._overlays["minimal_contour"].enabled
    object_list.select_item(object_list.item(0))
    _process_events()
    assert viewer_qt_viewer(viewer).controls.widgets[labels].mc_button.isChecked()


def test_minimal_contour_roi_is_the_bounding_box_of_the_selected_object(viewer, scene, monkeypatch):
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import \
        VispyMinimalContourOverlay
    from napari_nd_annotator.minimal_contour import roi as mc_roi
    # the planes of the scene are tiny: a small margin keeps the ROI clearly smaller than the plane
    monkeypatch.setattr(mc_roi, "MARGIN_PIXELS", 2)
    image, labels, object_list = scene
    annotator = _annotator(viewer)
    viewer.layers.selection.select_only(labels)
    mc_widget = annotator.ToolsWidget.minimal_contour_widget
    viewer_qt_viewer(viewer).controls.widgets[labels].mc_button.click()
    _process_events()
    assert mc_widget.image_layer is image and labels._overlays["minimal_contour"].enabled
    assert VispyMinimalContourOverlay._plane_shape is None  # no object: the whole plane

    _select_object(viewer, labels, object_list)
    # no new plane was visited: the selection alone makes the contour work on the object
    box = mc_roi.object_box(BBOX_MIN, BBOX_MAX, (1, 2), image.data.shape[1:])
    assert box is not None and box.area < image.data.shape[1] * image.data.shape[2]
    assert VispyMinimalContourOverlay._roi_origin == box.origin
    assert VispyMinimalContourOverlay._features_shape == box.shape
    assert VispyMinimalContourOverlay._plane_shape == image.data.shape[1:]

    # another bounding box: the ROI follows it
    shift = np.array([0, 3, 2])
    _move_bounding_box(object_list, shift)
    moved = mc_roi.object_box(BBOX_MIN + shift, BBOX_MAX + shift, (1, 2), image.data.shape[1:])
    assert VispyMinimalContourOverlay._roi_origin == moved.origin and moved.origin != box.origin

    object_list.select_item(object_list.item(0))
    _process_events()
    assert VispyMinimalContourOverlay._plane_shape is None and VispyMinimalContourOverlay._roi_origin == (0, 0)


def test_selection_notifies_the_listeners(viewer, scene):
    _, labels, object_list = scene
    selections = []

    def listener(event):
        if event.viewer is viewer:
            selected = selected_object(viewer)
            selections.append(None if selected is None else (selected[0], tuple(selected[1])))

    selected_object_changed.connect(listener)
    try:
        _select_object(viewer, labels, object_list)
        assert selections[-1] == (OBJECT_ID, tuple(BBOX_MIN))
        _move_bounding_box(object_list, np.array([1, 2, 3]))
        assert selections[-1] == (OBJECT_ID, tuple(BBOX_MIN + [1, 2, 3]))
        object_list.select_item(object_list.item(0))
        assert selections[-1] is None
    finally:
        selected_object_changed.disconnect(listener)


def test_icon_follows_the_edits_of_the_selected_object(viewer, scene):
    _, labels, object_list = scene
    _select_object(viewer, labels, object_list)
    item = object_list.item(0)
    before = item._icon.copy()
    # the middle slice of the object is shown in the icon
    labels.data[4, BBOX_MIN[1]:BBOX_MIN[1] + 8, BBOX_MIN[2]:BBOX_MIN[2] + 8] = OBJECT_ID
    labels.refresh()
    _wait()
    assert not np.array_equal(item._icon, before)
    # no longer watched once the object is unselected
    object_list.select_item(item)
    after = item._icon.copy()
    labels.data[4] = 0
    labels.refresh()
    _wait()
    np.testing.assert_array_equal(item._icon, after)


def test_create_layers_action_adds_cropped_views_that_line_up_with_the_originals(viewer, scene):
    image, labels, object_list = scene
    crop_image, crop_labels = object_list.item(0).create_layers()
    assert crop_labels.metadata["source_image_layer"] is crop_image
    for original, crop in ((image, crop_image), (labels, crop_labels)):
        np.testing.assert_allclose(crop.scale, original.scale)
        np.testing.assert_array_equal(crop.data.shape, BBOX_MAX - BBOX_MIN + 1)
        # first and last voxel of the crop are at the same world position as the bbox corners in the original
        np.testing.assert_allclose(crop.data_to_world((0, 0, 0)), original.data_to_world(BBOX_MIN))
        np.testing.assert_allclose(crop.data_to_world(np.array(crop.data.shape) - 1),
                                   original.data_to_world(BBOX_MAX))
    # an explicit action: selecting objects does not do it
    assert object_list.selected_idx is None


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


def test_deleting_bounding_boxes_updates_the_object_list(viewer, scene):
    _, _, object_list = scene
    _add_objects(object_list, [5, 6])
    _delete_box(object_list, 1)
    assert [object_list.item(i).idx for i in range(object_list.count())] == [OBJECT_ID, 6]
    _delete_box(object_list, 0)
    _delete_box(object_list, 0)
    assert object_list.count() == 0


def test_deleting_the_selected_objects_bounding_box_unselects_it(viewer, scene):
    _, labels, object_list = scene
    _add_objects(object_list, [5])
    before = _move_view_away(viewer, labels)
    layers = list(viewer.layers)
    _select_object(viewer, labels, object_list)
    assert object_list.projections_widget is not None
    _delete_box(object_list, 0)
    assert object_list.projections_widget is None
    assert object_list.selected_idx is None
    assert selected_object(viewer) is None
    _assert_view(viewer, before)
    assert [layer for layer in viewer.layers if layer not in layers and not layer.name.startswith("Cropped")] == []
    assert not [layer for layer in viewer.layers if layer.name.startswith("Cropped")]
    assert [object_list.item(i).idx for i in range(object_list.count())] == [5]


def test_deleting_another_bounding_box_keeps_the_object_selected(viewer, scene):
    _, labels, object_list = scene
    _add_objects(object_list, [5])
    object_list.selected_idx = None
    viewer.layers.selection.select_only(labels)
    object_list.select_item(object_list.item(1))
    _process_events()
    assert object_list.selected_idx == 1
    _delete_box(object_list, 0)
    assert object_list.projections_widget is not None
    assert object_list.selected_idx == 0
    assert selected_object(viewer)[0] == 5


def test_replacing_the_labels_layer_of_a_selected_object_reselects_it(viewer, scene):
    image, labels, object_list = scene
    other = viewer.add_labels(np.zeros((10, 40, 40), dtype=np.uint16), name="other",
                              scale=SCALE, translate=TRANSLATE, metadata={"source_image_layer": image})
    _select_object(viewer, labels, object_list)
    first_projections = object_list.projections_widget
    object_list.mask_layer = other
    _process_events()
    assert object_list.selected_idx == 0
    assert object_list.projections_widget is not None and object_list.projections_widget is not first_projections
    assert object_list.projections_widget.mask_layer is other
    assert list(viewer.layers.selection) == [other] and other.selected_label == OBJECT_ID
