"""The labels layer as the one selection: its links (image, boxes), checked when they are made, and the object list
following the selected labels layer.

Like the other viewer tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools

import numpy as np
import pytest

pytest.importorskip("napari_bbox")

from .test_object_list import viewer, _process_events  # noqa: E402,F401
from napari_nd_annotator._widgets import working_set as ws  # noqa: E402

SHAPE = (6, 24, 24)


def _boxes_layer(viewer, name, with_box=True, scale=(1., 1., 1.)):
    from napari_bbox import BoundingBoxLayer
    layer = BoundingBoxLayer(ndim=3, name=name, edge_color="green", face_color="transparent", scale=scale)
    viewer.add_layer(layer)
    if with_box:
        corners = np.where(list(itertools.product((False, True), repeat=3)), (4, 18, 18), (1, 4, 4))
        layer.data = [corners.astype(float)]
        layer.features = {"label": np.array([3])}
    return layer


def _image(viewer, name, **kwargs):
    return viewer.add_image(np.zeros(SHAPE), name=name, **kwargs)


def _labels(viewer, name, shape=SHAPE, **kwargs):
    return viewer.add_labels(np.zeros(shape, dtype=np.uint16), name=name, **kwargs)


def _select(viewer, layer):
    viewer.layers.selection.select_only(layer)
    _process_events()


# --- the links, checked when they are made

def test_the_source_image_must_fit_the_labels_layer(viewer):
    image, labels = _image(viewer, "image"), _labels(viewer, "labels")
    ws.set_source_image(labels, image, viewer)
    assert ws.linked_image(labels, viewer) is image
    wrong_shape = viewer.add_image(np.zeros((6, 24, 20)), name="wrong shape")
    with pytest.raises(ws.LinkError, match="different shapes"):
        ws.set_source_image(labels, wrong_shape, viewer)
    flat = viewer.add_image(np.zeros((24, 24)), name="flat")
    with pytest.raises(ws.LinkError):
        ws.set_source_image(labels, flat, viewer)
    assert ws.linked_image(labels, viewer) is image  # a refused link changes nothing


def test_the_labels_layer_and_its_boxes_take_the_scale_of_the_image(viewer):
    image = _image(viewer, "image", scale=(3., 1., 1.))
    labels = _labels(viewer, "labels")
    boxes = _boxes_layer(viewer, "boxes", with_box=False)
    ws.set_source_image(labels, image, viewer)
    ws.set_boxes(labels, boxes, viewer)
    assert tuple(labels.scale) == (3., 1., 1.) and tuple(boxes.scale) == (3., 1., 1.)
    assert boxes.metadata["source_image_layer"] is image  # the boxes belong to the image of the labels layer


def test_the_boxes_need_the_dimensions_of_the_labels_layer(viewer):
    from napari_bbox import BoundingBoxLayer
    labels = _labels(viewer, "labels")
    flat_boxes = BoundingBoxLayer(ndim=2, name="flat boxes", edge_color="green", face_color="transparent")
    viewer.add_layer(flat_boxes)
    with pytest.raises(ws.LinkError, match="dimensions"):
        ws.set_boxes(labels, flat_boxes, viewer)
    assert ws.linked_boxes(labels, viewer) is None


def test_the_only_fitting_image_is_linked_automatically(viewer, monkeypatch):
    messages = []
    monkeypatch.setattr(ws, "show_info", messages.append)
    image = _image(viewer, "image")
    _image(viewer, "other shape").data = np.zeros((6, 24, 20))
    labels = _labels(viewer, "labels")
    assert ws.auto_link(viewer, labels) is image
    assert ws.linked_image(labels, viewer) is image and len(messages) == 1 and "'image'" in messages[0]
    assert ws.auto_link(viewer, labels) is image and len(messages) == 1  # linked already: nothing is said


def test_several_fitting_images_are_not_guessed_between(viewer):
    _image(viewer, "first")
    _image(viewer, "second")
    labels = _labels(viewer, "labels")
    assert ws.auto_link(viewer, labels) is None
    assert "source_image_layer" not in labels.metadata


def test_an_image_of_another_scale_is_not_linked_automatically(viewer):
    _image(viewer, "image", scale=(3., 1., 1.))
    labels = _labels(viewer, "labels")
    assert ws.auto_link(viewer, labels) is None


def test_the_active_working_set_is_that_of_a_labels_layer_or_its_boxes_only(viewer):
    image, labels, boxes = _image(viewer, "image"), _labels(viewer, "labels"), _boxes_layer(viewer, "boxes")
    points = viewer.add_points(np.zeros((1, 3)), name="points")
    ws.link_layers(image=image, labels=labels, boxes=boxes)
    for layer in (labels, boxes):
        _select(viewer, layer)
        result = ws.active_working_set(viewer)
        assert (result.image, result.labels, result.boxes) == (image, labels, boxes)
    for layer in (image, points):
        _select(viewer, layer)
        assert ws.active_working_set(viewer) is None
    stray = _boxes_layer(viewer, "stray boxes")  # no labels layer refers to it
    _select(viewer, stray)
    assert ws.active_working_set(viewer) is None


# --- the object list follows the selected labels layer

@pytest.fixture
def two_sets(viewer):
    """Two images, each with a labels layer and a box layer with one object, and an object list."""
    from napari_nd_annotator._widgets.object_list import ListWidgetBB
    layers = {}
    for name in ("a", "b"):
        image, labels = _image(viewer, "image " + name), _labels(viewer, "labels " + name)
        boxes = _boxes_layer(viewer, "boxes " + name)
        ws.link_layers(image=image, labels=labels, boxes=boxes)
        layers[name] = (image, labels, boxes)
    viewer.layers.selection.clear()  # (the list follows the selected labels layer, there is none yet)
    widget = ListWidgetBB(viewer)
    viewer.window.add_dock_widget(widget, name="Object list")
    _process_events()
    return layers, widget


def test_without_a_choice_the_list_does_not_pick_one_of_several_layers(two_sets):
    layers, widget = two_sets
    assert widget.image.layer is None and widget.labels.layer is None and widget.bounding_box.layer is None
    assert widget.list_widget is None


def test_a_single_candidate_is_taken(viewer):
    from napari_nd_annotator._widgets.object_list import ListWidgetBB
    image, labels, boxes = _image(viewer, "image"), _labels(viewer, "labels"), _boxes_layer(viewer, "boxes")
    widget = ListWidgetBB(viewer)
    viewer.window.add_dock_widget(widget, name="Object list")
    assert (widget.image.layer, widget.labels.layer, widget.bounding_box.layer) == (image, labels, boxes)


def test_selecting_a_labels_layer_sets_the_list_to_its_image_and_boxes(viewer, two_sets):
    layers, widget = two_sets
    for name in ("a", "b", "a"):
        image, labels, boxes = layers[name]
        _select(viewer, labels)
        assert (widget.image.layer, widget.labels.layer, widget.bounding_box.layer) == (image, labels, boxes)
        assert widget.list_widget.bounding_box_layer is boxes and widget.list_widget.count() == 1
        assert widget.list_widget.image_layer is image and widget.list_widget.mask_layer is labels


def test_the_selected_object_is_unselected_when_the_labels_layer_changes(viewer, two_sets):
    from napari_nd_annotator._widgets.object_list import selected_object_info
    layers, widget = two_sets
    _select(viewer, layers["a"][1])
    widget.list_widget.select_item(widget.list_widget.item(0))
    _process_events()
    assert selected_object_info(viewer).labels_layer is layers["a"][1]
    _select(viewer, layers["b"][1])
    assert selected_object_info(viewer) is None
    # the highlight of the box of the first set went with the selection
    assert not np.allclose(layers["a"][2].edge_color[0], (1., 0., 0., 1.))
    widget.list_widget.select_item(widget.list_widget.item(0))
    _process_events()
    assert selected_object_info(viewer).labels_layer is layers["b"][1]


def test_selecting_the_boxes_of_the_list_keeps_the_list(viewer, two_sets):
    from napari_nd_annotator._widgets.object_list import selected_object_info
    layers, widget = two_sets
    _select(viewer, layers["a"][1])
    widget.list_widget.select_item(widget.list_widget.item(0))
    _select(viewer, layers["a"][2])  # to draw boxes
    assert selected_object_info(viewer).bbox_layer is layers["a"][2]
    assert widget.list_widget.bounding_box_layer is layers["a"][2]


def test_a_helper_layer_or_an_image_leaves_the_list_as_it_is(viewer, two_sets):
    layers, widget = two_sets
    _select(viewer, layers["a"][1])
    _select(viewer, layers["b"][0])  # an image
    _select(viewer, viewer.add_points(np.zeros((1, 3)), name="points"))
    assert widget.labels.layer is layers["a"][1] and widget.image.layer is layers["a"][0]


def test_labels_without_boxes_give_the_list_without_objects_until_boxes_are_created(viewer, two_sets):
    layers, widget = two_sets
    image = _image(viewer, "image c")  # an image that has no boxes yet
    labels = _labels(viewer, "new labels", metadata={"source_image_layer": image})
    _select(viewer, labels)
    assert widget.labels.layer is labels and widget.image.layer is image
    assert widget.bounding_box.layer is None and widget.list_widget is None
    widget.create_bb_layer(False)
    _process_events()
    boxes = widget.bounding_box.layer
    assert boxes is not None and boxes not in (layers["a"][2], layers["b"][2]) and widget.list_widget is not None
    assert ws.linked_boxes(labels, viewer) is boxes  # the new layer belongs to the labels layer


def test_two_labels_layers_can_share_the_boxes(viewer, two_sets):
    layers, widget = two_sets
    image, labels, boxes = layers["a"]
    second = _labels(viewer, "second labels", metadata={"source_image_layer": image, "object_boxes_layer": boxes})
    _select(viewer, layers["b"][1])
    _select(viewer, second)
    assert widget.labels.layer is second and widget.bounding_box.layer is boxes
    assert widget.list_widget.mask_layer is second and widget.list_widget.count() == 1


def test_a_labels_layer_without_a_source_image_leaves_the_list_as_it_is(viewer, two_sets, monkeypatch):
    layers, widget = two_sets
    monkeypatch.setattr(ws, "show_info", lambda message: None)
    _select(viewer, layers["a"][1])
    stray = _labels(viewer, "stray")  # two images fit it: none is guessed
    _select(viewer, stray)
    assert widget.labels.layer is layers["a"][1] and widget.list_widget.bounding_box_layer is layers["a"][2]


def test_the_layers_chosen_in_the_list_are_linked(viewer):
    from napari_nd_annotator._widgets.object_list import ListWidgetBB
    image, other = _image(viewer, "image"), _image(viewer, "other")
    labels = _labels(viewer, "labels", metadata={"source_image_layer": other})
    boxes = _boxes_layer(viewer, "boxes")
    second_boxes = _boxes_layer(viewer, "second boxes")
    widget = ListWidgetBB(viewer)
    viewer.window.add_dock_widget(widget, name="Object list")
    widget.labels.layer = labels
    widget.image.layer = image
    widget.bounding_box.layer = boxes
    assert ws.linked_image(labels, viewer) is image and ws.linked_boxes(labels, viewer) is boxes
    widget.bounding_box.layer = second_boxes
    assert ws.linked_boxes(labels, viewer) is second_boxes


def test_removing_a_layer_of_the_list_does_not_switch_to_another_one(viewer, two_sets):
    layers, widget = two_sets
    _select(viewer, layers["a"][1])
    viewer.layers.remove(layers["a"][0])
    _process_events()
    assert widget.image.layer is None  # (there is another image: it is not taken instead)
    assert widget.labels.layer is layers["a"][1]


def test_the_list_that_was_replaced_leaves_nothing_behind(viewer, two_sets):
    layers, widget = two_sets
    _select(viewer, layers["a"][1])
    old_list = widget.list_widget
    callbacks = layers["a"][2].mouse_drag_callbacks
    assert old_list.bounding_box_change in callbacks
    _select(viewer, layers["b"][1])
    assert widget.list_widget is not old_list
    assert old_list.bounding_box_change not in callbacks
