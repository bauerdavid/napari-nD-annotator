"""The layers of the object list: what happens when one of them is missing or does not fit the others.

Like the other viewer tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools

import numpy as np
import pytest

pytest.importorskip("napari_bbox")

from .test_object_list import viewer, _process_events  # noqa: E402,F401


def _box_corners(lower, upper):
    return np.where(list(itertools.product((False, True), repeat=3)), upper, lower).astype(float)


@pytest.fixture
def widget(viewer):
    from napari_nd_annotator._widgets.object_list import ListWidgetBB
    widget = ListWidgetBB(viewer)
    viewer.window.add_dock_widget(widget, name="Object list")
    _process_events()
    return widget


def test_creating_a_bounding_box_layer_needs_an_image(viewer, widget):
    with pytest.warns(UserWarning, match="image layer"):
        widget.create_bb_layer(False)
    assert len(viewer.layers) == 0


def test_boxes_from_labels_need_an_image(viewer, widget):
    viewer.add_labels(np.ones((5, 20, 20), dtype=np.uint16), name="labels")
    _process_events()
    assert widget.image_layer is None and widget.mask_layer is not None
    with pytest.warns(UserWarning, match="image layer"):
        widget.bounding_boxes_from_labels()
    assert [layer.name for layer in viewer.layers] == ["labels"]


def test_boxes_from_labels_need_labels(viewer, widget):
    viewer.add_image(np.zeros((5, 20, 20)), name="image")
    _process_events()
    with pytest.warns(UserWarning, match="labels layer"):
        widget.bounding_boxes_from_labels()
    assert [layer.name for layer in viewer.layers] == ["image"]


def test_selecting_an_object_of_mismatched_layers_says_so_once(viewer, monkeypatch):
    from napari_bbox import BoundingBoxLayer
    from napari_nd_annotator._widgets import object_list as object_list_module
    messages = []
    monkeypatch.setattr(object_list_module, "show_info", messages.append)
    image = viewer.add_image(np.zeros((10, 40, 40)), name="image")
    labels = viewer.add_labels(np.zeros((10, 40, 30), dtype=np.uint16), name="labels")
    bbox_layer = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(bbox_layer)
    bbox_layer.data = [_box_corners((2, 5, 5), (6, 20, 20))]
    bbox_layer.features = {"label": np.array([1])}
    object_list = object_list_module.ObjectListWidget(viewer, "Object", bbox_layer, image, labels, None, 2)
    _process_events()
    for _ in range(3):  # select, unselect, select
        object_list.select_item(object_list.item(0))
        _process_events()
    assert len(messages) == 1
    assert "labels" in messages[0] and "image" in messages[0] and "different shapes" in messages[0]


def test_matching_layers_are_not_reported(viewer, monkeypatch):
    from napari_bbox import BoundingBoxLayer
    from napari_nd_annotator._widgets import object_list as object_list_module
    messages = []
    monkeypatch.setattr(object_list_module, "show_info", messages.append)
    image = viewer.add_image(np.zeros((10, 40, 40)), name="image")
    labels = viewer.add_labels(np.zeros((10, 40, 40), dtype=np.uint16), name="labels")
    bbox_layer = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(bbox_layer)
    bbox_layer.data = [_box_corners((2, 5, 5), (6, 20, 20))]
    bbox_layer.features = {"label": np.array([1])}
    object_list = object_list_module.ObjectListWidget(viewer, "Object", bbox_layer, image, labels, None, 2)
    _process_events()
    object_list.select_item(object_list.item(0))
    _process_events()
    assert messages == []


def _list_with_box_layer(viewer, with_box=True):
    from napari_bbox import BoundingBoxLayer
    from napari_nd_annotator._widgets.object_list import ObjectListWidget
    image = viewer.add_image(np.zeros((10, 40, 40)), name="image")
    labels = viewer.add_labels(np.zeros((10, 40, 40), dtype=np.uint16), name="labels")
    bbox_layer = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(bbox_layer)
    if with_box:
        bbox_layer.data = [_box_corners((2, 5, 5), (6, 20, 20))]
        bbox_layer.features = {"label": np.array([1])}
    object_list = ObjectListWidget(viewer, "Object", bbox_layer, image, labels, None, 2)
    _process_events()
    return object_list, bbox_layer


def _finish_drawing(object_list, layer, new_data=None):
    """A press and a release on the box layer; ``new_data`` is what the drawing made of the layer's data."""
    from types import SimpleNamespace
    gen = object_list.bounding_box_change(layer, SimpleNamespace(type="mouse_release"))
    next(gen)  # the press
    if new_data is not None:
        layer.data = new_data
    try:
        next(gen)  # the release
    except StopIteration:
        pass
    _process_events()


def test_a_box_drawn_outside_the_image_is_dropped(viewer):
    object_list, bbox_layer = _list_with_box_layer(viewer, with_box=False)
    _finish_drawing(object_list, bbox_layer, [_box_corners((100, 100, 100), (110, 110, 110))])
    assert len(bbox_layer.data) == 0 and object_list.count() == 0


def test_a_box_partly_outside_the_image_is_clipped(viewer):
    object_list, bbox_layer = _list_with_box_layer(viewer, with_box=False)
    _finish_drawing(object_list, bbox_layer, [_box_corners((2, 30, 30), (6, 60, 60))])
    assert len(bbox_layer.data) == 1
    assert np.max(bbox_layer.data[0]) <= 39


def test_releasing_the_mouse_on_a_layer_without_boxes(viewer):
    object_list, bbox_layer = _list_with_box_layer(viewer, with_box=False)
    _finish_drawing(object_list, bbox_layer)
    assert object_list.count() == 0


def test_the_previous_box_layer_no_longer_reaches_the_list(viewer):
    from napari_bbox import BoundingBoxLayer
    object_list, first_layer = _list_with_box_layer(viewer)
    assert object_list.bounding_box_change in first_layer.mouse_drag_callbacks
    second_layer = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(second_layer)
    object_list.bounding_box_layer = second_layer
    assert object_list.bounding_box_change not in first_layer.mouse_drag_callbacks
    assert object_list.bounding_box_change in second_layer.mouse_drag_callbacks
    assert object_list._on_bb_double_click not in first_layer.mouse_double_click_callbacks
