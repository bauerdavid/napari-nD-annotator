"""The optional box around the selected label of Labels layers, and its row in the layer controls."""
import numpy as np
import pytest

from napari_nd_annotator._helper_functions import viewer_qt_viewer
from napari_nd_annotator._widgets.selected_label_bbox import (
    OVERLAY_KEY, SelectedLabelBBoxOverlay, install_selected_label_bbox,
)
from napari_nd_annotator._widgets.selected_label_bbox.vispy_selected_label_bbox_overlay import (
    box_vertices, label_bounds,
)


def _process_events():
    from qtpy.QtWidgets import QApplication
    QApplication.processEvents()


# --- geometry, without a viewer ---------------------------------------------------------

def test_label_bounds_are_inclusive_pixel_indices():
    data = np.zeros((10, 20), dtype=np.uint8)
    data[2:5, 7:15] = 3
    data[8, 1] = 4
    np.testing.assert_array_equal(label_bounds(data, 3), [[2, 4], [7, 14]])
    np.testing.assert_array_equal(label_bounds(data, 4), [[8, 8], [1, 1]])
    assert label_bounds(data, 5) is None


def test_label_bounds_3d():
    data = np.zeros((6, 7, 8), dtype=np.uint16)
    data[1, 2, 3] = 9
    data[4, 5, 6] = 9
    np.testing.assert_array_equal(label_bounds(data, 9), [[1, 4], [2, 5], [3, 6]])


def test_box_vertices_cover_whole_pixels_in_vispy_order():
    vertices = box_vertices(np.array([[2, 4], [7, 14]]))  # rows, columns
    # vispy order is (x, y): columns first
    np.testing.assert_array_equal(vertices.min(axis=0), [6.5, 1.5])
    np.testing.assert_array_equal(vertices.max(axis=0), [14.5, 4.5])
    assert len(vertices) == 4
    assert len(box_vertices(np.array([[0, 1], [0, 1], [0, 1]]))) == 8


# --- in a viewer -------------------------------------------------------------------------

@pytest.fixture
def viewer():
    import napari
    viewer = napari.Viewer(show=False)
    install_selected_label_bbox(viewer)
    yield viewer
    viewer.close()
    _process_events()


@pytest.fixture
def labels(viewer):
    data = np.zeros((5, 30, 40), dtype=np.uint8)
    data[1:4, 5:10, 20:31] = 1
    data[2, 15:18, 2:4] = 2
    layer = viewer.add_labels(data)
    _process_events()
    return layer


def _controls(viewer, layer):
    return viewer_qt_viewer(viewer).controls.widgets[layer].selected_label_bbox_controls


def _vispy_overlay(viewer, layer):
    """The overlay's visual; None where napari creates it lazily and the overlay was never shown."""
    canvas = viewer_qt_viewer(viewer).canvas
    overlay = layer._overlays[OVERLAY_KEY]
    visual_map = getattr(canvas, "_layer_overlay_to_visual", None)
    if visual_map is not None:
        return visual_map[layer].get(overlay)
    return canvas.layer_to_visual[layer].overlays[overlay]


def _shown_extent(viewer, layer):
    """(min, max) of the drawn box in numpy axis order of the displayed dims, or None if hidden."""
    vispy_overlay = _vispy_overlay(viewer, layer)
    if vispy_overlay is None or not vispy_overlay.node.visible:
        return None
    pos = vispy_overlay.node.pos[:, :len(layer._slice_input.displayed)][:, ::-1]
    return pos.min(axis=0).tolist(), pos.max(axis=0).tolist()


def test_every_labels_layer_gets_the_overlay_and_a_controls_row(viewer, labels):
    overlay = labels._overlays[OVERLAY_KEY]
    assert isinstance(overlay, SelectedLabelBBoxOverlay)
    assert not overlay.visible
    controls = viewer_qt_viewer(viewer).controls.widgets[labels]
    row_widget = controls.selected_label_bbox_controls
    assert controls.layout().indexOf(row_widget) >= 0
    assert not row_widget.checkbox.isChecked()

    # a second call does not add a second row
    install_selected_label_bbox(viewer)
    n_rows = controls.layout().rowCount()
    from napari_nd_annotator._widgets.selected_label_bbox.controls import add_bbox_controls
    assert add_bbox_controls(viewer, labels)
    assert controls.layout().rowCount() == n_rows


def test_existing_layers_get_the_overlay_on_install():
    import napari
    viewer = napari.Viewer(show=False)
    try:
        layer = viewer.add_labels(np.zeros((10, 10), dtype=np.uint8))
        image = viewer.add_image(np.zeros((10, 10)))
        install_selected_label_bbox(viewer)
        assert OVERLAY_KEY in layer._overlays
        assert OVERLAY_KEY not in image._overlays
        assert _controls(viewer, layer) is not None
    finally:
        viewer.close()
        _process_events()


def test_checkbox_and_overlay_stay_in_sync(viewer, labels):
    row = _controls(viewer, labels)
    overlay = labels._overlays[OVERLAY_KEY]
    row.checkbox.setChecked(True)
    assert overlay.visible
    overlay.visible = False
    assert not row.checkbox.isChecked()


def test_color_swatch_and_overlay_stay_in_sync(viewer, labels):
    row = _controls(viewer, labels)
    overlay = labels._overlays[OVERLAY_KEY]
    row.color_swatch.setColor("cyan")
    np.testing.assert_allclose(overlay.color, [0, 1, 1, 1])
    overlay.color = "magenta"
    np.testing.assert_allclose(row.color_swatch.color, [1, 0, 1, 1])
    labels._overlays[OVERLAY_KEY].visible = True
    np.testing.assert_allclose(_vispy_overlay(viewer, labels).node.color, [1, 0, 1, 1])


def _wait_for(condition, timeout=2.):
    """Process Qt events until ``condition()`` holds (the overlay recomputes on a timer after some changes)."""
    import time
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        _process_events()
        time.sleep(.01)
    return condition()


def test_box_follows_the_selected_label_in_2d(viewer, labels):
    viewer.dims.set_current_step(0, 2)
    labels.selected_label = 1
    assert _shown_extent(viewer, labels) is None  # off by default
    _controls(viewer, labels).checkbox.setChecked(True)
    assert _shown_extent(viewer, labels) == ([4.5, 19.5], [9.5, 30.5])

    labels.selected_label = 2
    assert _shown_extent(viewer, labels) == ([14.5, 1.5], [17.5, 3.5])

    labels.selected_label = labels.colormap.background_value
    assert _shown_extent(viewer, labels) is None
    labels.selected_label = 7  # not in the layer
    assert _shown_extent(viewer, labels) is None


def test_2d_box_is_the_projection_of_the_annotated_slices(viewer):
    data = np.zeros((7, 30, 40), dtype=np.uint8)
    data[1, 2:4, 3:6] = 5
    data[5, 20:25, 30:35] = 5
    layer = viewer.add_labels(data)
    layer.selected_label = 5
    layer._overlays[OVERLAY_KEY].visible = True
    expected = ([1.5, 2.5], [24.5, 34.5])
    for z in range(7):
        viewer.dims.set_current_step(0, z)
        _process_events()
        # also on the slices between the annotated ones, but not outside them
        assert _shown_extent(viewer, layer) == (expected if 1 <= z <= 5 else None), z


def test_3d_box_is_the_projection_of_the_other_dims(viewer):
    data = np.zeros((3, 5, 30, 40), dtype=np.uint8)  # t, z, y, x
    data[0, 1, 2:4, 3:6] = 5
    data[2, 3, 20:25, 30:35] = 5
    layer = viewer.add_labels(data)
    layer.selected_label = 5
    layer._overlays[OVERLAY_KEY].visible = True
    viewer.dims.ndisplay = 3
    viewer.dims.set_current_step(0, 1)
    _process_events()
    assert _shown_extent(viewer, layer) == ([0.5, 1.5, 2.5], [3.5, 24.5, 34.5])

    layer.data[2] = 0
    layer.events.paint(value=[])
    assert _shown_extent(viewer, layer) is None  # t = 1 is now after the label's last time point


def test_box_follows_painting(viewer, labels):
    viewer.dims.set_current_step(0, 0)  # label 2 is only on slice 2
    labels.selected_label = 2
    labels._overlays[OVERLAY_KEY].visible = True
    assert _shown_extent(viewer, labels) is None
    labels.brush_size = 1
    labels.paint((0, 25, 35), 2, refresh=True)
    _process_events()
    labels.events.paint(value=[])  # what ending a stroke sends
    assert _shown_extent(viewer, labels) == ([14.5, 1.5], [25.5, 35.5])


def test_box_follows_data_written_without_paint_events(viewer, labels):
    """The plugin's tools write into the data and refresh the layer."""
    labels.selected_label = 2
    labels._overlays[OVERLAY_KEY].visible = True
    labels.data[4, 29, 39] = 2
    labels.refresh()
    assert _wait_for(lambda: _shown_extent(viewer, labels) == ([14.5, 1.5], [29.5, 39.5]))

    new_data = np.zeros_like(labels.data)
    new_data[viewer.dims.current_step[0], 0, 0] = 2
    labels.data = new_data
    assert _shown_extent(viewer, labels) == ([-.5, -.5], [.5, .5])


def test_box_is_3d_in_3d_view(viewer, labels):
    labels.selected_label = 1
    labels._overlays[OVERLAY_KEY].visible = True
    viewer.dims.ndisplay = 3
    _process_events()
    assert _shown_extent(viewer, labels) == ([0.5, 4.5, 19.5], [3.5, 9.5, 30.5])
    assert len(_vispy_overlay(viewer, labels).node.pos) == 8


def test_box_follows_transposed_dims(viewer, labels):
    labels.selected_label = 1
    labels._overlays[OVERLAY_KEY].visible = True
    viewer.dims.order = (1, 0, 2)  # displays (z, x)
    viewer.dims.set_current_step(1, 7)  # label 1 spans y = 5..9
    _process_events()
    assert _shown_extent(viewer, labels) == ([0.5, 19.5], [3.5, 30.5])
    viewer.dims.set_current_step(1, 0)
    _process_events()
    assert _shown_extent(viewer, labels) is None


def test_hidden_layer_hides_the_box(viewer, labels):
    viewer.dims.set_current_step(0, 2)
    labels.selected_label = 1
    labels._overlays[OVERLAY_KEY].visible = True
    labels.visible = False
    assert not _vispy_overlay(viewer, labels).node.visible
    labels.visible = True
    assert _shown_extent(viewer, labels) is not None
