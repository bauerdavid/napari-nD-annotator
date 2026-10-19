"""The orthogonal projections of the selected object follow the cropped layers, whatever tool is active.

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
    _, labels, object_list = scene
    crop_image, crop_labels = _select_object(viewer, labels, object_list)
    widget = object_list.projections_widget
    assert widget is not None
    _flush(widget)
    return crop_image, crop_labels, object_list, widget


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
    crop_image, crop_labels, object_list, widget = selected
    _resize_bounding_box(object_list, new_max)
    new_shape = tuple(new_max - BBOX_MIN + 1)
    assert crop_image.data.shape == new_shape
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
    crop_image, crop_labels, object_list, widget = selected
    viewer.cursor.position = crop_labels.data_to_world(np.array(crop_labels.data.shape) - 1)
    _flush(widget)
    _resize_bounding_box(object_list, BBOX_MAX - [3, 8, 5])
    _flush(widget)
    for projection in widget.projections:
        assert all(0 <= s < size for s, size in zip(projection.slices, crop_image.data.shape))
        projection.update()  # no "images do not match"


def test_mask_and_image_of_different_sizes_do_not_crash(selected):
    crop_image, crop_labels, _, widget = selected
    # the layers are resized one after the other
    crop_image.data = np.zeros(np.array(crop_image.data.shape) + 3, dtype=crop_image.data.dtype)
    for projection in widget.projections:
        projection.update()
    _flush(widget)


@pytest.mark.parametrize("mode", [Mode.PAN_ZOOM, Mode.FILL, Mode.PICK])
def test_projections_follow_a_programmatic_labels_write(viewer, selected, mode):
    _, crop_labels, _, widget = selected
    crop_labels.mode = mode
    before = _images(widget)
    crop_labels.data[...] = OBJECT_ID
    crop_labels.refresh()
    _flush(widget)
    assert all(old != new for old, new in zip(before, _images(widget)))

    before = _images(widget)
    crop_labels.data = np.zeros_like(crop_labels.data)
    _flush(widget)
    assert all(old != new for old, new in zip(before, _images(widget)))


def test_projections_follow_a_stroke(selected):
    _, crop_labels, _, widget = selected
    crop_labels.mode = Mode.PAN_ZOOM
    updates = []
    crop_labels.events.labels_update.connect(updates.append)
    before = _images(widget)
    # the projections show the planes of dimensions (0, 1), (0, 2) and (1, 2): the voxel is in all of them
    center = [widget.projections[2].slices[0], widget.projections[1].slices[1], widget.projections[0].slices[2]]
    for _ in range(3):
        crop_labels.paint(center, OBJECT_ID)
    assert updates, "napari does not emit labels_update for a stroke"
    _flush(widget)
    assert all(old != new for old, new in zip(before, _images(widget)))


def test_a_stroke_is_not_rendered_for_every_event(selected):
    _, crop_labels, _, widget = selected
    counts = _count_updates(widget)
    for _ in range(50):
        crop_labels.events.set_data()
        crop_labels.events.data(value=crop_labels.data)
    _flush(widget)
    assert counts == [1] * len(widget.projections)


@pytest.mark.parametrize("mode", [Mode.PAN_ZOOM, Mode.PAINT, Mode.FILL])
def test_crosshair_follows_the_cursor_position(viewer, selected, mode):
    crop_image, crop_labels, _, widget = selected
    # the active layer and the tool do not matter
    crop_labels.mode = mode
    viewer.layers.selection.select_only(crop_image)
    for point in ((2, 7, 9), (3, 4, 14)):
        viewer.cursor.position = crop_labels.data_to_world(point)
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
    _, crop_labels, _, widget = selected
    crop_labels.mode = Mode.PAN_ZOOM
    viewer.cursor.position = crop_labels.data_to_world((2, 7, 9))
    _flush(widget)
    for projection in widget.projections:
        if tuple(sorted(projection.displayed_axes)) != (1, 2):
            assert projection._overlay.sum() > 0
            assert projection._overlay.sum() <= max(projection._overlay.shape)


def test_cursor_is_ignored_in_3d(viewer, selected):
    _, crop_labels, _, widget = selected
    viewer.cursor.position = crop_labels.data_to_world((2, 7, 9))
    _flush(widget)
    slices = [list(projection.slices) for projection in widget.projections]
    viewer.dims.ndisplay = 3
    viewer.cursor.position = crop_labels.data_to_world((1, 3, 4))
    _flush(widget)
    assert [list(projection.slices) for projection in widget.projections] == slices
    viewer.dims.ndisplay = 2


def _nothing_happens(viewer, crop_labels, widget):
    counts = _count_updates(widget)
    viewer.cursor.position = crop_labels.data_to_world((1, 3, 4))
    crop_labels.events.set_data()
    crop_labels.events.data(value=crop_labels.data)
    _flush(widget)
    return counts == [0] * len(widget.projections)


def test_no_connections_are_left_after_closing(viewer, selected):
    _, crop_labels, _, widget = selected
    assert not _nothing_happens(viewer, crop_labels, widget)
    widget.close()
    _process_events()
    assert widget._connections == []
    assert _nothing_happens(viewer, crop_labels, widget)


def test_no_connections_are_left_after_the_layers_are_removed(viewer, selected):
    _, crop_labels, _, widget = selected
    viewer.layers.remove(crop_labels)
    _process_events()
    assert widget._connections == []
    assert _nothing_happens(viewer, crop_labels, widget)


def test_unselecting_the_object_disconnects_the_projections(viewer, selected):
    _, crop_labels, object_list, widget = selected
    object_list.item(0).unselect()
    _process_events()
    assert object_list.projections_widget is None
    assert widget._connections == []
    assert _nothing_happens(viewer, crop_labels, widget)


def test_projections_can_be_reconnected(viewer, selected):
    _, crop_labels, _, widget = selected
    widget._disconnect_events()
    assert _nothing_happens(viewer, crop_labels, widget)
    widget._connect_events()
    assert not _nothing_happens(viewer, crop_labels, widget)


def _viewer_slice(viewer, layer, dim):
    """Index of the slice of the layer that the viewer position falls in."""
    return int(round(layer.world_to_data(viewer.dims.point)[dim]))


def test_projections_show_the_slice_visible_in_the_viewer(viewer, selected):
    crop_image, crop_labels, _, widget = selected
    # the scene is anisotropic and translated, so the viewer position is not the index of the slice
    for z in (1, 3, 4):
        viewer.dims.set_point(0, crop_labels.data_to_world((z, 0, 0))[0])
        _flush(widget)
        assert _viewer_slice(viewer, crop_labels, 0) == z
        assert widget.projections[2].slices[0] == z  # the projection of dimensions (1, 2)


def test_clicking_a_projection_shows_its_slice_in_the_viewer(viewer, selected):
    crop_image, crop_labels, _, widget = selected
    viewer.cursor.position = crop_labels.data_to_world((2, 7, 9))
    _flush(widget)
    projection = widget.projections[0]  # dimensions (0, 1) at the slice of dimension 2
    projection.on_click(None)
    _process_events()
    assert sorted(viewer.dims.displayed) == [0, 1]
    assert projection.slices[2] == 9
    assert _viewer_slice(viewer, crop_labels, 2) == _viewer_slice(viewer, crop_image, 2) == 9
