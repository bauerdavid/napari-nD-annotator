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
