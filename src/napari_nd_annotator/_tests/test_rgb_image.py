"""Minimal Surface on an RGB image: the user chooses the channel it works on, and a failure is always shown.

Like the other viewer tests these need a GL-capable Qt platform, the optional minimal_surface package and the
napari-bbox plugin.
"""
import itertools

import numpy as np
import pytest

pytest.importorskip("minimal_surface")
pytest.importorskip("napari_bbox")

from .test_napari_compat import viewer, _process_events  # noqa: E402,F401
from .test_suggest_anchors_widget import (  # noqa: E402,F401
    annotator, _select, _anchors, _distance_from_ball_surface, BALL_CENTER, BALL_RADIUS, SHAPE)

LABEL = 3


def _shown(widget, parent):
    """Whether the widget is not hidden itself (the toolbox is not shown in the tests)."""
    return widget.native.isVisibleTo(parent.native)


@pytest.fixture
def rgb_scene(viewer, annotator):
    """An RGB image of a ball that is only in the red channel (green: noise, blue: nothing), with an object around it."""
    from napari_bbox import BoundingBoxLayer
    from napari_nd_annotator._widgets.object_list import ObjectListWidget
    from scipy.ndimage import gaussian_filter
    zz, yy, xx = np.mgrid[tuple(slice(0, s) for s in SHAPE)]
    distance = np.sqrt(((np.stack([zz, yy, xx], -1) - BALL_CENTER) ** 2).sum(-1))
    rng = np.random.default_rng(0)
    red = gaussian_filter((distance <= BALL_RADIUS).astype(float), 1) * 200 + rng.normal(0, 4, SHAPE)
    green = rng.normal(100, 4, SHAPE)
    rgb = np.stack([red, green, np.zeros(SHAPE)], axis=-1).clip(0, 255).astype(np.uint8)
    image_layer = viewer.add_image(rgb, name="rgb ball", rgb=True)
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels",
                               metadata={"source_image_layer": image_layer})
    bbox_min, bbox_max = BALL_CENTER - 15, BALL_CENTER + 15
    corners = np.where(list(itertools.product((False, True), repeat=3)), bbox_max, bbox_min)
    bbox_layer = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(bbox_layer)
    bbox_layer.data = [corners.astype(float)]
    bbox_layer.features = {"label": np.array([LABEL])}
    object_list = ObjectListWidget(viewer, "Object", bbox_layer, image_layer, labels, None, 4)
    _process_events()
    ms = annotator.ToolsWidget.minimal_surface_widget
    ms.image_layer = image_layer
    _select(viewer, object_list)
    return image_layer, rgb, object_list, ms


def test_the_channel_can_be_chosen_for_an_rgb_image(rgb_scene):
    image_layer, rgb, object_list, ms = rgb_scene
    assert image_layer.rgb and image_layer.ndim == 3
    assert _shown(ms.channel_combobox, ms)
    assert "RGB" in ms.image_status_label.value
    assert list(ms.channel_combobox.choices) == ["Gray (mean of R, G, B)", "Red", "Green", "Blue"]
    # the buttons are not disabled: an RGB image is not a reason
    assert ms.suggest_anchors_button.enabled


def test_the_channel_choice_is_hidden_for_other_images(viewer, annotator):
    ms = annotator.ToolsWidget.minimal_surface_widget
    layer = viewer.add_image(np.zeros(SHAPE), name="gray")
    ms.image_layer = layer
    _process_events()
    assert not _shown(ms.channel_combobox, ms)
    # it appears with an RGB image and goes away again
    rgb_layer = viewer.add_image(np.zeros(SHAPE + (3,), dtype=np.uint8), name="rgb", rgb=True)
    ms.image_layer = rgb_layer
    _process_events()
    assert _shown(ms.channel_combobox, ms)
    ms.image_layer = layer
    _process_events()
    assert not _shown(ms.channel_combobox, ms)


@pytest.mark.parametrize("channel, index", [("Red", 0), ("Green", 1), ("Blue", 2)])
def test_the_intensity_is_the_chosen_channel(rgb_scene, channel, index):
    image_layer, rgb, object_list, ms = rgb_scene
    ms.channel_combobox.value = channel
    data = ms._intensity_data(image_layer)
    assert data.shape == SHAPE
    np.testing.assert_array_equal(data, rgb[..., index])


def test_the_mean_of_the_channels_is_computed_once(rgb_scene):
    image_layer, rgb, object_list, ms = rgb_scene
    ms.channel_combobox.value = "Gray (mean of R, G, B)"
    mean = ms._intensity_data(image_layer)
    assert mean.shape == SHAPE and mean.dtype == np.float32
    np.testing.assert_allclose(mean, rgb[..., :3].mean(axis=-1), rtol=1e-5)
    assert ms._intensity_data(image_layer) is mean


def test_anchors_are_suggested_on_the_chosen_channel(rgb_scene):
    image_layer, rgb, object_list, ms = rgb_scene
    ms.channel_combobox.value = "Red"
    ms._suggest_anchors()
    points = _anchors(ms)
    assert points.shape == (2, 3)
    assert np.all(_distance_from_ball_surface(points) < 2.5)
    assert np.linalg.norm(points[0] - points[1]) > 1.8 * BALL_RADIUS


def test_a_failing_suggestion_is_reported_every_time(rgb_scene):
    """The blue channel has nothing in it: the user is told so, again when pressing the button again."""
    image_layer, rgb, object_list, ms = rgb_scene
    ms.channel_combobox.value = "Blue"
    for _ in range(3):
        with pytest.warns(UserWarning, match="No anchor suggestion"):
            ms._suggest_anchors()
    assert len(_anchors(ms)) == 0


def test_a_run_gets_the_chosen_channel(rgb_scene, monkeypatch):
    from napari_nd_annotator._widgets import minimal_surface_widget as msw
    image_layer, rgb, object_list, ms = rgb_scene
    ms.channel_combobox.value = "Red"
    ms._suggest_anchors()
    received = {}

    class Stop(Exception):
        pass

    def fake_initialize(self, image_data, *args, **kwargs):
        received["image"] = np.asarray(image_data)
        raise Stop

    monkeypatch.setattr(msw.EstimationWorker, "initialize", fake_initialize)
    with pytest.raises(Stop):
        ms._start_estimation()
    np.testing.assert_array_equal(received["image"], rgb[..., 0])


def test_anchors_can_be_placed_on_an_rgb_image(rgb_scene):
    """Placing anchors in the 3D view is for the selected object whatever kind of image it has."""
    from types import SimpleNamespace
    image_layer, rgb, object_list, ms = rgb_scene
    ms.channel_combobox.value = "Red"
    ms._viewer.dims.ndisplay = 3
    ms.native.window()
    event = SimpleNamespace(type="mouse_press", button=1, modifiers=(), position=(20, 30, -20),
                            view_direction=(0, 0, 1), dims_displayed=(0, 1, 2))
    point = ms._pick_anchor(ms._selected_object(), event.position, event.view_direction, event.dims_displayed)
    assert point is not None
    assert abs(point[2] - (BALL_CENTER[2] - BALL_RADIUS)) < 1.5
