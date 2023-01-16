import numpy as np
from ..callbacks import (
    extend_mask,
    reduce_mask,
    increment_selected_label,
    decrement_selected_label,
    scroll_to_prev,
    scroll_to_next,
    increase_brush_size,
    decrease_brush_size
)


def test_extend_mask(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 2, 0, 0, 0],
            [0, 0, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 0, 0],
            [0, 0, 0, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    expected_output = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 2, 0, 0, 0],
            [0, 1, 2, 2, 2, 2, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 2, 0],
            [0, 0, 2, 2, 2, 2, 2, 0],
            [0, 0, 0, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    labels_layer = viewer.add_labels(data)
    labels_layer.selected_label = 2
    extend_mask(labels_layer)
    assert np.allclose(labels_layer.data, expected_output)


def test_reduce_mask(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 2, 0, 0, 0],
            [0, 0, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 0, 0],
            [0, 0, 0, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    expected_output = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    labels_layer = viewer.add_labels(data)
    labels_layer.selected_label = 2
    reduce_mask(labels_layer)
    assert np.allclose(labels_layer.data, expected_output)


def test_increment_selected_label(make_napari_viewer):
    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(np.zeros((100, 100), dtype=int))
    prev_selected_label = labels_layer.selected_label
    increment_selected_label(labels_layer)
    assert labels_layer.selected_label == prev_selected_label + 1


def test_decrement_selected_label(make_napari_viewer):
    viewer = make_napari_viewer()
    original_selected_label = 2
    labels_layer = viewer.add_labels(np.zeros((100, 100), dtype=int))
    labels_layer.selected_label = original_selected_label
    decrement_selected_label(labels_layer)
    assert labels_layer.selected_label == original_selected_label - 1


def test_decrement_selected_label_non_negative(make_napari_viewer):
    viewer = make_napari_viewer()
    original_selected_label = 0
    labels_layer = viewer.add_labels(np.zeros((100, 100), dtype=int))
    labels_layer.selected_label = original_selected_label
    decrement_selected_label(labels_layer)
    assert labels_layer.selected_label == original_selected_label


def test_scroll_to_prev(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((100, 100, 100), dtype=int))
    prev_step = 10
    viewer.dims.set_current_step(viewer.dims.not_displayed[0], prev_step)
    scroll_to_prev_cb = scroll_to_prev(viewer)
    scroll_to_prev_cb(None)
    assert viewer.dims.current_step[viewer.dims.not_displayed[0]] == prev_step - 1


def test_scroll_to_next(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((100, 100, 100), dtype=int))
    prev_step = 10
    viewer.dims.set_current_step(viewer.dims.not_displayed[0], prev_step)
    scroll_to_next_cb = scroll_to_next(viewer)
    scroll_to_next_cb(None)
    assert viewer.dims.current_step[viewer.dims.not_displayed[0]] == prev_step + 1


def test_increase_brush_size(make_napari_viewer):
    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(np.zeros((100, 100, 100), dtype=int))
    prev_brush_size = 10
    labels_layer.brush_size = prev_brush_size
    increase_brush_size(labels_layer)
    assert labels_layer.brush_size > prev_brush_size


def test_decrease_brush_size(make_napari_viewer):
    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(np.zeros((100, 100, 100), dtype=int))
    prev_brush_size = 10
    labels_layer.selected_label = prev_brush_size
    decrease_brush_size(labels_layer)
    assert labels_layer.brush_size < prev_brush_size
