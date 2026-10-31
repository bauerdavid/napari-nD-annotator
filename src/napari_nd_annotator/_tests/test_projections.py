"""The orthogonal projections of the selected object show its bounding box of the original layers, whatever tool is active.

Like the object list tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools
import time

import numpy as np
import pytest
from napari.layers.labels._labels_constants import Mode

pytest.importorskip("napari_bbox")

from napari_nd_annotator._tests.test_object_list import (  # noqa: E402, F401
    BBOX_MAX, BBOX_MIN, OBJECT_ID, SCALE, TRANSLATE, _process_events, _select_object, scene, viewer)


@pytest.fixture
def selected(viewer, scene):
    image, labels, object_list = scene
    _select_object(viewer, labels, object_list)
    widget = object_list.projections_widget
    assert widget is not None
    _flush(widget)
    return image, labels, object_list, widget


def _world(layer, box_point):
    """World position of a point given relative to the first corner of the bounding box (the projections' indices)."""
    return layer.data_to_world(np.asarray(box_point) + BBOX_MIN)


def _flush(widget):
    """Lets the throttled refresh of the projections happen."""
    from qtpy.QtWidgets import QApplication
    deadline = time.time() + 2
    while time.time() < deadline:
        QApplication.processEvents()
        if not widget._refresh_timer.isActive():
            break
        time.sleep(0.005)
    QApplication.processEvents()


def _count_updates(widget):
    counts = [0] * len(widget.projections)
    for i, projection in enumerate(widget.projections):
        original = projection.update

        def counting_update(*args, _i=i, _original=original, **kwargs):
            counts[_i] += 1
            return _original(*args, **kwargs)

        projection.update = counting_update
    return counts


def _images(widget):
    return [projection.img.copy() for projection in widget.projections]


def _resize_bounding_box(object_list, new_max):
    corners = np.where(list(itertools.product((False, True), repeat=3)), new_max, BBOX_MIN)
    object_list.bounding_box_layer.data = [(corners*SCALE + TRANSLATE).astype(float)]
    object_list.update_items()
    _process_events()


@pytest.mark.parametrize("new_max", [BBOX_MAX + [2, 6, 4], BBOX_MAX - [3, 8, 5]])
def test_projections_survive_a_resized_bounding_box(selected, new_max):
    image, labels, object_list, widget = selected
    _resize_bounding_box(object_list, new_max)
    new_shape = tuple(new_max - BBOX_MIN + 1)
    assert widget.data_shape == new_shape
    # what the object list does after changing a bounding box, and what the dimension sliders and the
    # dims events do on their own
    for projection in widget.projections:
        projection.update_overlay()
        projection.update()
    for projection in widget.projections:
        projection.update()
    _flush(widget)
    for projection in widget.projections:
        assert projection._overlay.shape == projection.overlay_shape
        assert all(0 <= s < size for s, size in zip(projection.slices, new_shape))
    assert [slider.maximum() for slider in widget.sliders] == [size - 1 for size in new_shape]


def test_resized_bounding_box_moves_slices_and_crosshair_into_range(viewer, selected):
    image, labels, object_list, widget = selected
    viewer.cursor.position = _world(labels, BBOX_MAX - BBOX_MIN)
    _flush(widget)
    _resize_bounding_box(object_list, BBOX_MAX - [3, 8, 5])
    _flush(widget)
    for projection in widget.projections:
        assert all(0 <= s < size for s, size in zip(projection.slices, widget.data_shape))
        projection.update()  # no "images do not match"


def test_mask_and_image_of_different_sizes_do_not_crash(selected):
    image, labels, _, widget = selected
    # the layers are replaced one after the other: the new image does not reach the end of the bounding box
    image.data = np.zeros((10, 20, 20), dtype=image.data.dtype)
    for projection in widget.projections:
        projection.update()
    _flush(widget)


@pytest.mark.parametrize("mode", [Mode.PAN_ZOOM, Mode.FILL, Mode.PICK])
def test_projections_follow_a_programmatic_labels_write(viewer, selected, mode):
    _, labels, _, widget = selected
    labels.mode = mode
    before = _images(widget)
    labels.data[...] = OBJECT_ID
    labels.refresh()
    _flush(widget)
    assert all(old != new for old, new in zip(before, _images(widget)))

    before = _images(widget)
    labels.data = np.zeros_like(labels.data)
    _flush(widget)
    assert all(old != new for old, new in zip(before, _images(widget)))


def test_projections_follow_a_stroke(selected):
    _, labels, _, widget = selected
    labels.mode = Mode.PAN_ZOOM
    updates = []
    labels.events.labels_update.connect(updates.append)
    before = _images(widget)
    # the projections show the planes of dimensions (0, 1), (0, 2) and (1, 2): the voxel is in all of them
    center = [widget.projections[2].slices[0], widget.projections[1].slices[1], widget.projections[0].slices[2]]
    center = list(np.asarray(center) + BBOX_MIN)
    for _ in range(3):
        labels.paint(center, OBJECT_ID)
    assert updates, "napari does not emit labels_update for a stroke"
    _flush(widget)
    assert all(old != new for old, new in zip(before, _images(widget)))


def test_a_stroke_is_not_rendered_for_every_event(selected):
    _, labels, _, widget = selected
    counts = _count_updates(widget)
    for _ in range(50):
        labels.events.set_data()
        labels.events.data(value=labels.data)
    _flush(widget)
    assert counts == [1] * len(widget.projections)


@pytest.mark.parametrize("mode", [Mode.PAN_ZOOM, Mode.PAINT, Mode.FILL])
def test_crosshair_follows_the_cursor_position(viewer, selected, mode):
    image, labels, _, widget = selected
    # the active layer and the tool do not matter
    labels.mode = mode
    viewer.layers.selection.select_only(image)
    for point in ((2, 7, 9), (3, 4, 14)):
        viewer.cursor.position = _world(labels, point)
        _flush(widget)
        for projection in widget.projections:
            assert projection.slices[1] == point[1] and projection.slices[2] == point[2]
        projection = next(p for p in widget.projections if tuple(sorted(p.displayed_axes)) == (1, 2))
        expected = np.zeros(projection._overlay.shape, bool)
        expected[point[1]] = True
        expected[:, point[2]] = True
        np.testing.assert_array_equal(projection._overlay, expected)
        assert [slider.value() for slider in widget.sliders[1:]] == list(point[1:])


def test_orthogonal_projections_show_the_cursor_without_a_brush(viewer, selected):
    _, labels, _, widget = selected
    labels.mode = Mode.PAN_ZOOM
    viewer.cursor.position = _world(labels, (2, 7, 9))
    _flush(widget)
    for projection in widget.projections:
        if tuple(sorted(projection.displayed_axes)) != (1, 2):
            assert projection._overlay.sum() > 0
            assert projection._overlay.sum() <= max(projection._overlay.shape)


def test_cursor_is_ignored_in_3d(viewer, selected):
    _, labels, _, widget = selected
    viewer.cursor.position = _world(labels, (2, 7, 9))
    _flush(widget)
    slices = [list(projection.slices) for projection in widget.projections]
    viewer.dims.ndisplay = 3
    viewer.cursor.position = _world(labels, (1, 3, 4))
    _flush(widget)
    assert [list(projection.slices) for projection in widget.projections] == slices
    viewer.dims.ndisplay = 2


def _nothing_happens(viewer, labels, widget):
    counts = _count_updates(widget)
    viewer.cursor.position = _world(labels, (1, 3, 4))
    labels.events.set_data()
    labels.events.data(value=labels.data)
    _flush(widget)
    return counts == [0] * len(widget.projections)


def test_no_connections_are_left_after_closing(viewer, selected):
    _, labels, _, widget = selected
    assert not _nothing_happens(viewer, labels, widget)
    widget.close()
    _process_events()
    assert widget._connections == []
    assert _nothing_happens(viewer, labels, widget)


def test_no_connections_are_left_after_the_layers_are_removed(viewer, selected):
    _, labels, _, widget = selected
    viewer.layers.remove(labels)
    _process_events()
    assert widget._connections == []
    assert _nothing_happens(viewer, labels, widget)


def test_unselecting_the_object_disconnects_the_projections(viewer, selected):
    _, labels, object_list, widget = selected
    object_list.item(0).unselect()
    _process_events()
    assert object_list.projections_widget is None
    assert widget._connections == []
    assert _nothing_happens(viewer, labels, widget)


def test_projections_can_be_reconnected(viewer, selected):
    _, labels, _, widget = selected
    widget._disconnect_events()
    assert _nothing_happens(viewer, labels, widget)
    widget._connect_events()
    assert not _nothing_happens(viewer, labels, widget)


def _viewer_slice(viewer, layer, dim):
    """Index of the slice of the layer that the viewer position falls in."""
    return int(round(layer.world_to_data(viewer.dims.point)[dim]))


def test_projections_show_the_slice_visible_in_the_viewer(viewer, selected):
    image, labels, _, widget = selected
    # the scene is anisotropic and translated, so the viewer position is not the index of the slice
    for z in (1, 3, 4):
        viewer.dims.set_point(0, _world(labels, (z, 0, 0))[0])
        _flush(widget)
        assert _viewer_slice(viewer, labels, 0) == z + BBOX_MIN[0]
        assert widget.projections[2].slices[0] == z  # the projection of dimensions (1, 2)


def test_clicking_a_projection_shows_its_slice_in_the_viewer(viewer, selected):
    image, labels, _, widget = selected
    viewer.cursor.position = _world(labels, (2, 7, 9))
    _flush(widget)
    projection = widget.projections[0]  # dimensions (0, 1) at the slice of dimension 2
    projection.on_click(None)
    _process_events()
    assert sorted(viewer.dims.displayed) == [0, 1]
    assert projection.slices[2] == 9
    assert _viewer_slice(viewer, labels, 2) == _viewer_slice(viewer, image, 2) == 9 + BBOX_MIN[2]


def test_projections_work_on_the_original_layers_and_show_the_bounding_box_only(viewer, selected):
    image, labels, object_list, widget = selected
    assert widget.image_layer is image and widget.mask_layer is labels
    assert widget.data_shape == tuple(BBOX_MAX - BBOX_MIN + 1)
    for projection in widget.projections:
        assert np.shares_memory(projection.image_data, image.data)
        assert np.shares_memory(projection.mask_data, labels.data)
        assert projection.image_data.shape == widget.data_shape
    assert [slider.maximum() for slider in widget.sliders] == [size - 1 for size in widget.data_shape]


def test_projections_show_what_is_painted_inside_the_bounding_box_only(viewer, selected):
    image, labels, _, widget = selected
    before = _images(widget)
    labels.data[BBOX_MAX[0] + 1:, :5, :5] = OBJECT_ID  # outside of the bounding box
    labels.refresh()
    _flush(widget)
    assert _images(widget) == before
    labels.data[BBOX_MIN[0]:BBOX_MIN[0] + 2, BBOX_MIN[1]:BBOX_MIN[1] + 3, BBOX_MIN[2]:BBOX_MIN[2] + 3] = OBJECT_ID
    labels.refresh()
    _flush(widget)
    assert _images(widget) != before


def test_projections_follow_a_moved_bounding_box(viewer, selected):
    image, labels, object_list, widget = selected
    shift = np.array([1, 2, 3])
    corners = np.where(list(itertools.product((False, True), repeat=3)), BBOX_MAX + shift, BBOX_MIN + shift)
    object_list.bounding_box_layer.data = [(corners*SCALE + TRANSLATE).astype(float)]
    object_list.update_items()
    _process_events()
    _flush(widget)
    np.testing.assert_array_equal(widget.box.min, BBOX_MIN + shift)
    assert widget.data_shape == tuple(BBOX_MAX - BBOX_MIN + 1)
    for projection in widget.projections:
        np.testing.assert_array_equal(projection.image_data, image.data[object_list.item(0).bbox_idx])
