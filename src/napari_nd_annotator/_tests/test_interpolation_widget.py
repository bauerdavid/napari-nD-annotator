import time
import numpy as np
import napari
from qtpy.QtWidgets import QApplication
from napari_nd_annotator import AnnotatorWidget


def _process_events():
    for _ in range(20):
        QApplication.processEvents()


def test_labels_layer_with_fewer_slices_than_viewer_position():
    # Needs a GL-capable Qt platform (napari.Viewer), so run it without QT_QPA_PLATFORM=offscreen
    # (e.g. under xvfb with software GL on CI).
    viewer = napari.Viewer(show=False)
    try:
        widget = AnnotatorWidget(viewer, persist=False)
        viewer.window.add_dock_widget(widget, name="Annotator")
        interpolation_widget = widget.ToolsWidget.interpolation_widget
        viewer.add_image(np.random.default_rng(0).random((40, 20, 20)))
        viewer.dims.set_point(0, 20)

        # the viewer is past the last slice of this layer; adding it makes it the active layer
        small = viewer.add_labels(np.zeros((10, 10, 10), dtype=np.uint16))
        _process_events()
        assert interpolation_widget.active_labels_layer is small
        assert interpolation_widget._contours_at(interpolation_widget.current_slice) == []

        # moving within and beyond the layer must not raise either
        for position in (3, 25, 9, -2):
            viewer.dims.set_point(0, position)
            _process_events()
            interpolation_widget._update_overlay()
    finally:
        viewer.close()


def test_interpolation_overlay_is_removed_when_a_mask_is_created_on_its_slice():
    # a mask that is not painted with the brush (e.g. a closed minimal contour polygon) goes through paint_polygon
    viewer = napari.Viewer(show=False)
    try:
        widget = AnnotatorWidget(viewer, persist=False)
        viewer.window.add_dock_widget(widget, name="Annotator")
        interpolation_widget = widget.ToolsWidget.interpolation_widget
        viewer.add_image(np.random.default_rng(0).random((30, 64, 64)))
        data = np.zeros((30, 64, 64), dtype=np.uint16)
        data[5, 20:40, 20:40] = 1
        data[15, 22:44, 22:44] = 1
        labels = viewer.add_labels(data)
        viewer.layers.selection.active = labels
        labels.selected_label = 1
        viewer.dims.set_point(0, 10)
        deadline = time.time() + 10
        overlay = labels._overlays["interpolation"]
        while len(overlay.contour) == 0 and time.time() < deadline:
            _process_events()
            time.sleep(0.02)
        assert len(overlay.contour) == 1  # interpolated between slices 5 and 15

        labels.paint_polygon(np.array([[10, 25, 25], [10, 25, 45], [10, 45, 45], [10, 45, 25]], float), 1)
        _process_events()

        assert len(overlay.contour) == 0
        assert interpolation_widget._contours_at(10) == []
    finally:
        # let the interpolation workers finish: they must not outlive the widgets they read from
        deadline = time.time() + 10
        while any(w is not None for w in interpolation_widget.workers) and time.time() < deadline:
            _process_events()
            time.sleep(0.02)
        viewer.close()


def test_painting_a_3d_labels_layer_in_the_3d_view():
    # With ndisplay=3 no dimension of a 3D layer is sliced, so the interpolation widget has no
    # dimension; its mouse_drag callback used to index with None and raise on every paint click,
    # which also stopped napari's own paint callback (registered after it) from running.
    from dataclasses import dataclass
    from napari.layers.labels._labels_mouse_bindings import draw
    from napari.utils._proxies import ReadOnlyWrapper
    from napari.utils.interactions import mouse_press_callbacks, mouse_release_callbacks

    @dataclass
    class MouseEvent:
        type: str
        position: tuple
        is_dragging: bool = False
        dims_displayed: tuple = (0, 1, 2)
        dims_point: tuple = (0, 0, 0)
        view_direction: tuple = (1, 0, 0)
        pos: tuple = (0, 0)
        button: int = 1
        modifiers: tuple = ()
        handled: bool = False

    viewer = napari.Viewer(show=False)
    try:
        widget = AnnotatorWidget(viewer, persist=False)
        viewer.window.add_dock_widget(widget, name="Annotator")
        interpolation_widget = widget.ToolsWidget.interpolation_widget
        data = np.zeros((20, 20, 20), dtype=np.uint16)
        data[8:12, 8:12, 8:12] = 1
        labels = viewer.add_labels(data)
        viewer.dims.ndisplay = 3
        _process_events()
        # switching to 3D may select the object list's ghost layer; paint on the real one
        viewer.layers.selection.active = labels
        _process_events()
        assert interpolation_widget.active_labels_layer is labels
        assert interpolation_widget.dimension is None

        labels.mode = "paint"
        labels.n_edit_dimensions = 3
        labels.brush_size = 3
        labels.selected_label = 2
        _process_events()
        callbacks = labels.mouse_drag_callbacks
        assert callbacks.index(interpolation_widget._on_mouse_drag) < callbacks.index(draw)

        # the ray along axis 0 through (10, 10) hits the object at (8, 10, 10)
        position = (0, 10, 10)
        mouse_press_callbacks(labels, ReadOnlyWrapper(MouseEvent(type="mouse_press", position=position)))
        mouse_release_callbacks(labels, MouseEvent(type="mouse_release", position=position))
        _process_events()

        assert labels.data[8, 10, 10] == 2
        assert not interpolation_widget._is_painting
    finally:
        deadline = time.time() + 10
        while any(w is not None for w in interpolation_widget.workers) and time.time() < deadline:
            _process_events()
            time.sleep(0.02)
        viewer.close()


# --- transforms and display order: the overlay shows the contours of the slice that is displayed, where it is

import pytest


def _wait_for(condition, timeout=10):
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        _process_events()
        time.sleep(0.02)
    _process_events()
    return condition()


def _wait_for_workers(interpolation_widget, timeout=10):
    # A recompute can still be queued (e.g. the autofill path of a paint stroke defers it with QTimer.singleShot):
    # its workers would start after the check, and yield contours into the widgets after the viewer is closed.
    _process_events()
    _wait_for(lambda: not interpolation_widget._recompute_pending
              and all(w is None for w in interpolation_widget.workers), timeout)


@pytest.fixture
def interpolation_scene():
    """A labels layer with the same wide rectangle on slices 5 and 15 (rows 10-19, columns 10-49); the slices in
    between get it interpolated."""
    viewer = napari.Viewer(show=False)
    widget = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(widget, name="Annotator")
    interpolation_widget = widget.ToolsWidget.interpolation_widget
    data = np.zeros((30, 64, 64), dtype=np.uint16)
    data[5, 10:20, 10:50] = 1
    data[15, 10:20, 10:50] = 1
    for z in range(30):  # a pixel that tells which slice is shown
        data[z, 63, 63] = 100 + z
    labels = viewer.add_labels(data)
    viewer.layers.selection.active = labels
    labels.selected_label = 1
    viewer.dims.set_point(0, 10)
    overlay = labels._overlays["interpolation"]
    assert _wait_for(lambda: len(overlay.contour) == 1)
    try:
        yield viewer, interpolation_widget, labels, overlay
    finally:
        _wait_for_workers(interpolation_widget)
        viewer.close()


def _shown_slice(labels):
    return int(np.asarray(labels._slice.image.raw)[63, 63]) - 100


def _vispy_overlay(viewer, labels, overlay):
    from napari_nd_annotator._helper_functions import viewer_qt_viewer
    return viewer_qt_viewer(viewer).canvas._layer_overlay_to_visual[labels][overlay]


def test_the_slice_of_the_widget_is_the_displayed_one_with_a_fractional_data_position(interpolation_scene):
    viewer, interpolation_widget, labels, overlay = interpolation_scene
    labels.translate = (.6, 0., 0.)
    for z in (4.6, 9.6, 10.6, 15.6):  # slider positions: their data positions are 3.9999.. and alike
        viewer.dims.set_point(0, z)
        _process_events()
        assert interpolation_widget.current_slice == _shown_slice(labels)
    labels.translate = (0., 0., 0.)
    labels.scale = (3., 1., 1.)
    for z in np.arange(0, 87, .5):
        viewer.dims.set_point(0, z)
        _process_events()
        assert interpolation_widget.current_slice == _shown_slice(labels), z


def test_the_overlay_follows_a_transform_change_that_shows_another_slice(interpolation_scene):
    viewer, interpolation_widget, labels, overlay = interpolation_scene
    # a layer of its own (not linked) keeps the range of the viewer, so the viewer does not move (not an image: the
    # minimal contour tool would ask whether it is the source image of the labels)
    viewer.add_points(np.array([[0., 0., 0.], [89., 7., 7.]]), name="unrelated")
    viewer.layers.selection.active = labels
    viewer.dims.set_point(0, 10)
    assert _wait_for(lambda: len(overlay.contour) == 1)
    labels.scale = (3., 1., 1.)  # the viewer stays at z=10: slice 3, which has no interpolated contour
    assert _shown_slice(labels) == 3
    assert _wait_for(lambda: len(overlay.contour) == 0)
    # (whichever slice is shown, its contours are)
    for scale, translate in ((1., 0.), (1., 4.), (1., 6.), (2., -6.), (1., 0.), (1., -5.)):  # slices 10 6 4 8 10 15
        labels.scale = (scale, 1., 1.)
        labels.translate = (translate, 0., 0.)
        shown = _shown_slice(labels)
        assert interpolation_widget.current_slice == shown
        expected = 1 if 5 < shown < 15 else 0  # interpolated between the annotated slices 5 and 15
        assert _wait_for(lambda: len(overlay.contour) == expected), (translate, shown)


def test_the_overlay_is_where_the_interpolated_slice_would_be_painted(interpolation_scene):
    viewer, interpolation_widget, labels, overlay = interpolation_scene
    for order, rows, cols in (((0, 1, 2), (10, 19), (10, 49)), ((0, 2, 1), (10, 49), (10, 19))):
        viewer.dims.order = order
        assert _wait_for(lambda: len(overlay.contour) == 1)
        _wait_for_workers(interpolation_widget)
        polygon = _vispy_overlay(viewer, labels, overlay)._polygons[0]
        # in the node of the layer: x along the second displayed dimension, y along the first
        x, y = np.asarray(polygon.pos)[:, 0], np.asarray(polygon.pos)[:, 1]
        assert (y.min(), y.max()) == rows and (x.min(), x.max()) == cols, order


def test_painting_bakes_the_interpolated_contour_when_slicing_along_another_axis():
    viewer = napari.Viewer(show=False)
    widget = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(widget, name="Annotator")
    interpolation_widget = widget.ToolsWidget.interpolation_widget
    try:
        data = np.zeros((64, 30, 64), dtype=np.uint16)
        data[10:20, 5, 10:50] = 1
        data[10:20, 15, 10:50] = 1
        labels = viewer.add_labels(data)
        viewer.layers.selection.active = labels
        labels.selected_label = 1
        viewer.dims.order = (1, 0, 2)  # slices along axis 1
        _process_events()
        viewer.dims.set_point(1, 10)
        overlay = labels._overlays["interpolation"]
        assert _wait_for(lambda: len(overlay.contour) == 1)
        assert interpolation_widget.dimension == 1

        class Event:
            type = "mouse_press"

        labels.mode = "paint"
        drag = interpolation_widget._on_mouse_drag(labels, Event())
        next(drag)
        painted = np.argwhere(labels.data == 1)
        on_slice = painted[painted[:, 1] == 10]
        np.testing.assert_array_equal(on_slice.min(0), (10, 10, 10))
        np.testing.assert_array_equal(on_slice.max(0), (19, 10, 49))
        assert set(np.unique(painted[:, 1])) == {5, 10, 15}
        Event.type = "mouse_release"
        with pytest.raises(StopIteration):
            next(drag)
    finally:
        _wait_for_workers(interpolation_widget)
        viewer.close()


def test_every_interpolated_contour_of_a_slice_is_drawn(interpolation_scene):
    viewer, interpolation_widget, labels, overlay = interpolation_scene
    labels.data[5, 30:40, 5:15] = 1
    labels.data[15, 30:40, 5:15] = 1
    interpolation_widget.method = "Distance-based"
    labels.refresh()
    assert _wait_for(lambda: len(overlay.contour) == 2)
    _wait_for_workers(interpolation_widget)
    vispy_overlay = _vispy_overlay(viewer, labels, overlay)
    assert len(vispy_overlay.node._subvisuals) == 2
    first, second = vispy_overlay._polygons
    assert first.visible and second.visible
    np.testing.assert_allclose(second.color.rgba, first.color.rgba)


# --- undo: Interpolate is one edit, undo/redo recompute the interpolation, a stroke undoes the contour it baked in

def _shown_has_label(labels, label=1):
    return bool(np.any(np.asarray(labels._slice.image.raw) == label))


def test_interpolate_is_one_edit_that_undo_reverts(interpolation_scene):
    viewer, interpolation_widget, labels, overlay = interpolation_scene
    _wait_for_workers(interpolation_widget)
    before = labels.data.copy()
    n_edits = len(labels._undo_history)
    interpolation_widget.Interpolate()
    assert _wait_for(lambda: interpolation_widget.interpolate_button.enabled and np.any(labels.data[10] == 1))
    interpolated = labels.data.copy()
    assert all(np.any(interpolated[z] == 1) for z in range(5, 16))
    assert len(labels._undo_history) == n_edits + 1
    assert _shown_has_label(labels) and len(overlay.contour) == 0

    labels.undo()
    np.testing.assert_array_equal(labels.data, before)
    assert not _shown_has_label(labels)
    # the interpolation is computed again: shown, and Interpolate applies it again
    assert _wait_for(lambda: len(overlay.contour) == 1)
    _wait_for_workers(interpolation_widget)
    labels.redo()
    np.testing.assert_array_equal(labels.data, interpolated)
    assert _shown_has_label(labels)
    _wait_for_workers(interpolation_widget)
    assert len(overlay.contour) == 0
    labels.undo()
    assert _wait_for(lambda: len(overlay.contour) == 1)
    _wait_for_workers(interpolation_widget)
    interpolation_widget.Interpolate()
    assert _wait_for(lambda: interpolation_widget.interpolate_button.enabled and np.any(labels.data[10] == 1))
    np.testing.assert_array_equal(labels.data, interpolated)


def test_undoing_an_edit_recomputes_the_interpolation(interpolation_scene):
    viewer, interpolation_widget, labels, overlay = interpolation_scene
    viewer.dims.set_point(0, 20)
    _wait_for_workers(interpolation_widget)
    labels.paint_polygon(np.array([[20, 10, 10], [20, 10, 49], [20, 19, 49], [20, 19, 10]], float), 1)
    _process_events()
    viewer.dims.set_point(0, 18)
    assert _wait_for(lambda: len(overlay.contour) == 1)  # between slices 15 and 20
    _wait_for_workers(interpolation_widget)
    labels.undo()
    assert not np.any(labels.data[20] == 1)
    assert _wait_for(lambda: len(overlay.contour) == 0)
    _wait_for_workers(interpolation_widget)
    assert len(overlay.contour) == 0
    labels.redo()
    assert _wait_for(lambda: len(overlay.contour) == 1)


def test_one_undo_reverts_a_stroke_and_the_interpolated_contour_it_painted(interpolation_scene):
    from dataclasses import dataclass
    from napari.utils._proxies import ReadOnlyWrapper
    from napari.utils.interactions import mouse_press_callbacks, mouse_release_callbacks

    @dataclass
    class MouseEvent:
        type: str
        position: tuple
        is_dragging: bool = False
        dims_displayed: tuple = (1, 2)
        dims_point: tuple = (10, 0, 0)
        view_direction: tuple = None
        pos: tuple = (0, 0)
        button: int = 1
        modifiers: tuple = ()
        handled: bool = False

    viewer, interpolation_widget, labels, overlay = interpolation_scene
    _wait_for_workers(interpolation_widget)
    before = labels.data.copy()
    n_edits = len(labels._undo_history)
    labels.mode = "paint"
    labels.brush_size = 3
    position = (10, 30, 30)  # outside the interpolated rectangle (rows 10-19)
    mouse_press_callbacks(labels, ReadOnlyWrapper(MouseEvent(type="mouse_press", position=position)))
    mouse_release_callbacks(labels, MouseEvent(type="mouse_release", position=position))
    _process_events()
    assert labels.data[10, 30, 30] == 1 and labels.data[10, 15, 30] == 1  # the stroke and the baked contour
    assert len(labels._undo_history) == n_edits + 1 and not labels._staged_history
    labels.undo()
    np.testing.assert_array_equal(labels.data, before)
