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
