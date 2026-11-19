"""The "Fit to object" button of the object list: a rough box on one slice becomes the box of the whole object."""
import itertools
import warnings

import numpy as np
import pytest

pytest.importorskip("napari_bbox")

from .test_object_list import viewer, _process_events  # noqa: E402,F401

SCALE = np.array([3., 1., 1.])
TRANSLATE = np.array([5., 2., 3.])
SHAPE = (40, 100, 100)
CENTER = np.array([20., 50., 50.])
RADII = np.array([10., 25., 25.])  # a ball of radius 25 in physical units
OBJECT_ID = 7


@pytest.fixture
def scene(viewer):
    """A bright ball in an anisotropic, translated image, a flat box on its middle slice, selected in the list."""
    from napari_bbox import BoundingBoxLayer
    from napari_nd_annotator._widgets.object_list import ListWidgetBB
    grid = np.mgrid[tuple(slice(0, s) for s in SHAPE)].astype(float)
    ball = sum(((g - c) / r) ** 2 for g, c, r in zip(grid, CENTER, RADII)) <= 1
    data = ball * 100. + 20 + np.random.default_rng(0).normal(0, 4, SHAPE)
    image = viewer.add_image(data, name="image", scale=SCALE, translate=TRANSLATE)
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels", scale=SCALE, translate=TRANSLATE,
                               metadata={"source_image_layer": image})
    bbox_layer = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(bbox_layer)
    widget = ListWidgetBB(viewer)
    viewer.window.add_dock_widget(widget, name="Object list")
    widget.bounding_box.layer = bbox_layer
    widget.image.layer = image
    widget.labels.layer = labels
    _process_events()
    return image, bbox_layer, widget


def _set_box(widget, bbox_layer, lower, upper):
    """The box (inclusive indices of the image) as the only object, with its label."""
    corners = np.where(list(itertools.product((False, True), repeat=3)), upper, lower) * SCALE + TRANSLATE
    bbox_layer.data = [corners.astype(float)]
    bbox_layer.features = {"label": np.array([OBJECT_ID])}
    widget.list_widget.update_items()
    widget.list_widget.select_item(widget.list_widget.item(0))
    _process_events()


def _box(widget):
    return widget.list_widget.item(0).bounding_box


def test_a_flat_rough_box_becomes_the_box_of_the_object(scene):
    image, bbox_layer, widget = scene
    _set_box(widget, bbox_layer, (20, 40, 42), (20, 60, 70))  # only the middle slice, off the middle
    widget.fit_selected_box()
    _process_events()
    lower, upper = CENTER - RADII, CENTER + RADII
    box = _box(widget)
    assert np.all(np.abs(box[0] - lower) <= 2), box
    assert np.all(np.abs(box[1] - upper) <= 2), box
    # the object keeps its label and is still the only one
    assert list(bbox_layer.features["label"]) == [OBJECT_ID]
    assert widget.list_widget.count() == 1 and widget.list_widget.item(0).idx == OBJECT_ID
    assert len(bbox_layer.data) == 1
    # the box layer holds the corners in its own coordinates: round trip through the image
    corners = np.asarray(bbox_layer.data[0])
    np.testing.assert_allclose(image.world_to_data(bbox_layer.data_to_world(corners.min(0))), box[0], atol=1e-6)
    np.testing.assert_allclose(image.world_to_data(bbox_layer.data_to_world(corners.max(0))), box[1], atol=1e-6)


def test_a_box_drawn_on_the_top_slice_grows_downwards(scene):
    image, bbox_layer, widget = scene
    top = int(CENTER[0] - RADII[0]) + 1
    _set_box(widget, bbox_layer, (top, 42, 42), (top, 58, 58))
    widget.fit_selected_box()
    _process_events()
    box = _box(widget)
    assert box[0][0] <= top and box[1][0] >= CENTER[0] + RADII[0] - 3
    assert np.all(box[1][1:] - box[0][1:] >= 2 * RADII[1:] - 4)


def test_a_box_on_the_background_is_left_alone_with_a_warning(scene):
    image, bbox_layer, widget = scene
    _set_box(widget, bbox_layer, (20, 2, 2), (20, 12, 12))
    before = _box(widget).copy()
    with pytest.warns(UserWarning, match="Cannot fit the box of object 7"):
        widget.fit_selected_box()
    np.testing.assert_array_equal(_box(widget), before)


def test_without_a_selected_object_nothing_happens(scene):
    image, bbox_layer, widget = scene
    with pytest.warns(UserWarning, match="Select an object"):
        widget.fit_selected_box()


def test_the_button_and_the_combobox_leave_the_keyboard_to_the_canvas(scene):
    from qtpy.QtCore import Qt
    _, _, widget = scene
    assert widget.fit_button.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert widget.fit_polarity_combobox.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert [widget.fit_polarity_combobox.itemText(i) for i in range(3)] == ["Auto", "Bright", "Dark"]


def test_the_polarity_can_be_chosen(scene):
    _, bbox_layer, widget = scene
    _set_box(widget, bbox_layer, (20, 40, 42), (20, 60, 70))
    widget.fit_polarity_combobox.setCurrentText("Dark")
    with pytest.warns(UserWarning, match="Cannot fit"):
        widget.fit_selected_box()
    widget.fit_polarity_combobox.setCurrentText("Bright")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        widget.fit_selected_box()
