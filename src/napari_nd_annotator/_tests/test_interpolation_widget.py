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
