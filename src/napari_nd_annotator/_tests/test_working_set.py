"""The working set: which image, labels and bounding box layers the tools work on (`_widgets/working_set.py`).

Like the other viewer tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools

import numpy as np
import pytest

pytest.importorskip("napari_bbox")

from .test_object_list import viewer, _process_events  # noqa: E402,F401
from napari_nd_annotator._widgets import working_set as ws  # noqa: E402

SHAPE = (6, 20, 20)


def _boxes(viewer, name="boxes"):
    from napari_bbox import BoundingBoxLayer
    layer = BoundingBoxLayer(ndim=3, name=name, edge_color="green", face_color="transparent")
    viewer.add_layer(layer)
    return layer


def _select(viewer, layer):
    viewer.layers.selection.select_only(layer)
    _process_events()


@pytest.fixture
def scene(viewer):
    """An image with its labels and box layer, linked, and another image and a points layer unrelated to them."""
    ws.install(viewer)
    image = viewer.add_image(np.zeros(SHAPE), name="image")
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels")
    boxes = _boxes(viewer)
    ws.link_layers(image=image, labels=labels, boxes=boxes)
    other = viewer.add_image(np.zeros(SHAPE), name="other")
    points = viewer.add_points(np.zeros((1, 3)), name="points")
    return viewer, image, labels, boxes, other, points


def test_the_links_are_written_in_one_place(viewer):
    image = viewer.add_image(np.zeros(SHAPE))
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16))
    boxes = _boxes(viewer)
    ws.link_layers(image=image, labels=labels, boxes=boxes)
    assert labels.metadata["source_image_layer"] is image and boxes.metadata["source_image_layer"] is image
    assert labels.metadata["object_boxes_layer"] is boxes
    assert ws.linked_image(labels, viewer) is image and ws.linked_boxes(labels, viewer) is boxes
    # a link to a layer that is not in the viewer is not a link
    viewer.layers.remove(image)
    assert ws.linked_image(labels) is image and ws.linked_image(labels, viewer) is None


def test_nothing_open_gives_an_empty_working_set(viewer):
    result = ws.working_set(viewer)
    assert result.is_empty and result.object is None


@pytest.mark.parametrize("which", ["labels", "boxes", "image"])
def test_any_layer_of_the_set_gives_the_whole_set(scene, which):
    viewer, image, labels, boxes, other, points = scene
    _select(viewer, {"labels": labels, "boxes": boxes, "image": image}[which])
    result = ws.working_set(viewer)
    assert (result.image, result.labels, result.boxes) == (image, labels, boxes)
    assert result.source == ws.ACTIVE_LAYER and result.object is None


def test_an_image_without_labels_has_a_set_of_its_own(scene):
    viewer, image, labels, boxes, other, points = scene
    _select(viewer, other)
    result = ws.working_set(viewer)
    assert result.image is other and result.labels is None and result.boxes is None


def test_several_labels_layers_of_an_image_are_not_guessed_between(scene):
    viewer, image, labels, boxes, other, points = scene
    second = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="second labels")
    ws.link_layers(image=image, labels=second)
    _select(viewer, image)
    assert ws.working_set(viewer).labels is None  # ambiguous
    _select(viewer, second)  # the user works on the second one: it is the one that was used with the image last
    assert ws.working_set(viewer).labels is second
    _select(viewer, image)
    result = ws.working_set(viewer)
    assert result.labels is second and result.image is image


def test_a_helper_layer_does_not_change_the_working_set(scene):
    viewer, image, labels, boxes, other, points = scene
    _select(viewer, labels)
    first = ws.working_set(viewer)
    _select(viewer, points)
    result = ws.working_set(viewer)
    assert (result.image, result.labels, result.boxes) == (first.image, first.labels, first.boxes)
    assert result.source == ws.LAST_USED
    # but not a set whose layers are gone
    viewer.layers.remove(labels)
    assert ws.working_set(viewer).labels is None


def test_links_to_removed_layers_are_dropped(scene):
    viewer, image, labels, boxes, other, points = scene
    viewer.layers.remove(image)
    assert "source_image_layer" not in labels.metadata and "source_image_layer" not in boxes.metadata
    viewer.layers.remove(boxes)
    assert "object_boxes_layer" not in labels.metadata
    _select(viewer, labels)
    result = ws.working_set(viewer)
    assert result.labels is labels and result.image is None and result.boxes is None


def test_the_selected_object_decides(viewer):
    from napari_nd_annotator._widgets.object_list import ObjectListWidget
    ws.install(viewer)
    image = viewer.add_image(np.zeros(SHAPE), name="image")
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels")
    boxes = _boxes(viewer)
    stray = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="stray labels")
    corners = np.where(list(itertools.product((False, True), repeat=3)), (4, 15, 15), (1, 3, 3))
    boxes.data = [corners.astype(float)]
    boxes.features = {"label": np.array([5])}
    object_list = ObjectListWidget(viewer, "Object", boxes, image, labels, None, 6)
    object_list.select_item(object_list.item(0))
    _process_events()
    # the object list links the pair it works on
    assert labels.metadata["source_image_layer"] is image and labels.metadata["object_boxes_layer"] is boxes
    _select(viewer, stray)
    result = ws.working_set(viewer)
    assert result.source == ws.SELECTED_OBJECT
    assert (result.image, result.labels, result.boxes) == (image, labels, boxes)
    assert result.object.label == 5
    # without the object the active layer decides again
    object_list.select_item(object_list.item(0))
    _process_events()
    assert ws.working_set(viewer).labels is stray


def test_the_object_list_repoints_the_source_image_of_its_labels(viewer):
    from napari_nd_annotator._widgets.object_list import ObjectListWidget
    ws.install(viewer)
    first = viewer.add_image(np.zeros(SHAPE), name="first")
    image = viewer.add_image(np.zeros(SHAPE), name="image")
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels", metadata={"source_image_layer": first})
    boxes = _boxes(viewer)
    corners = np.where(list(itertools.product((False, True), repeat=3)), (4, 15, 15), (1, 3, 3))
    boxes.data = [corners.astype(float)]
    boxes.features = {"label": np.array([5])}
    object_list = ObjectListWidget(viewer, "Object", boxes, image, labels, None, 6)
    object_list.select_item(object_list.item(0))
    _process_events()
    # the minimal contour tool (which reads the link) and the object list work on the same image
    assert labels.metadata["source_image_layer"] is image


def test_the_event_is_emitted_when_the_working_set_changes(scene):
    viewer, image, labels, boxes, other, points = scene
    seen = []
    listener = lambda event: seen.append(event.working_set)  # noqa: E731
    ws.working_set_changed.connect(listener)
    try:
        _select(viewer, labels)
        assert len(seen) == 1 and seen[-1].labels is labels
        _select(viewer, boxes)  # the same set
        assert len(seen) == 1
        _select(viewer, other)
        assert len(seen) == 2 and seen[-1].image is other
        _select(viewer, points)  # a helper layer: nothing changes
        assert len(seen) == 2
    finally:
        ws.working_set_changed.disconnect(listener)


def test_scratch_layers_are_not_part_of_a_working_set(scene):
    viewer, image, labels, boxes, other, points = scene
    _select(viewer, labels)
    before = ws.working_set(viewer)
    assert before.labels is labels
    slice_image = viewer.add_image(np.zeros((20, 20)), name="slice", metadata={ws.SCRATCH: True})
    slice_labels = viewer.add_labels(np.zeros((20, 20), dtype=int), name="slice labels", metadata={ws.SCRATCH: True})
    _select(viewer, slice_labels)
    assert ws.is_scratch(slice_labels) and not ws.is_scratch(labels)
    assert ws.active_working_set(viewer) is None
    after = ws.working_set(viewer)  # what was used last stays
    assert (after.image, after.labels, after.boxes) == (image, labels, boxes)
    _select(viewer, slice_image)
    assert ws.working_set(viewer).labels is labels
