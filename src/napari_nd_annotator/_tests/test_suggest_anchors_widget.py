"""The "Suggest anchors" button of the Minimal Surface widget.

Needs a GL-capable Qt platform, the optional minimal_surface package and the napari-bbox plugin (like the
other viewer tests, skipped when one is missing).
"""
import itertools
import warnings

import numpy as np
import pytest

pytest.importorskip("minimal_surface")
pytest.importorskip("napari_bbox")

from .test_napari_compat import viewer, _process_events  # noqa: E402,F401

BALL_CENTER = np.array([20., 30., 30.])
BALL_RADIUS = 12.
SHAPE = (40, 60, 60)


@pytest.fixture
def annotator(viewer):
    from napari_nd_annotator import AnnotatorWidget
    widget = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(widget, name="Annotator")
    yield widget


@pytest.fixture
def scene(viewer, annotator):
    """A noisy ball, a bounding box around it selected in the object list, the Minimal Surface widget on the ball."""
    from napari_bbox import BoundingBoxLayer
    from napari_nd_annotator._widgets.object_list import ObjectListWidget
    zz, yy, xx = np.mgrid[tuple(slice(0, s) for s in SHAPE)]
    distance = np.sqrt(((np.stack([zz, yy, xx], -1) - BALL_CENTER) ** 2).sum(-1))
    ball = (distance <= BALL_RADIUS).astype(float)
    from scipy.ndimage import gaussian_filter
    image = gaussian_filter(ball, 1) * 1000 + np.random.default_rng(0).normal(0, 20, SHAPE)
    image_layer = viewer.add_image(image, name="ball")
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels", metadata={"source_image_layer": image_layer})
    bbox_min, bbox_max = BALL_CENTER - 15, BALL_CENTER + 15
    corners = np.where(list(itertools.product((False, True), repeat=3)), bbox_max, bbox_min)
    bbox_layer = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(bbox_layer)
    bbox_layer.data = [corners.astype(float)]
    bbox_layer.features = {"label": np.array([3])}
    object_list = ObjectListWidget(viewer, "Object", bbox_layer, image_layer, labels, None, 4)
    _process_events()
    ms = annotator.ToolsWidget.minimal_surface_widget
    ms.image_layer = image_layer
    return image_layer, object_list, ms


def _select(viewer, object_list):
    object_list.select_item(object_list.item(0))
    _process_events()


def _distance_from_ball_surface(points):
    return np.abs(np.linalg.norm(np.asarray(points) - BALL_CENTER, axis=1) - BALL_RADIUS)


def test_suggests_anchors_for_the_selected_object(viewer, scene):
    image_layer, object_list, ms = scene
    _select(viewer, object_list)
    ms._suggest_anchors()
    points = np.asarray(ms.points_layer.data)
    assert points.shape == (2, 3)
    assert np.all(_distance_from_ball_surface(points) < 2.5)
    # roughly opposite each other
    assert np.linalg.norm(points[0] - points[1]) > 1.8 * BALL_RADIUS
    # Run uses the pair as it does one placed by hand
    assert len(ms.points_layer.data) % 2 == 0


def test_suggestion_works_on_the_cropped_layer(viewer, scene):
    image_layer, object_list, ms = scene
    _select(viewer, object_list)
    crop = object_list.crop_image_layer
    assert crop is not None
    ms.image_layer = crop
    ms._suggest_anchors()
    # like clicked points, the anchors are in world coordinates, which the cropped layer is translated to
    points = np.asarray(ms.points_layer.data)
    assert points.shape == (2, 3)
    assert np.all(_distance_from_ball_surface(points) < 2.5)


def test_suggesting_again_replaces_the_suggestion_but_keeps_user_anchors(viewer, scene):
    image_layer, object_list, ms = scene
    manual = np.array([[20., 30., 18.], [20., 30., 42.]])
    for point in manual:
        ms.points_layer.add(point)
    _select(viewer, object_list)
    ms._suggest_anchors()
    assert len(ms.points_layer.data) == 4
    np.testing.assert_allclose(ms.points_layer.data[:2], manual)
    ms._suggest_anchors()
    assert len(ms.points_layer.data) == 4
    np.testing.assert_allclose(ms.points_layer.data[:2], manual)
    # a suggestion that was moved is the user's: it stays
    moved = np.asarray(ms.points_layer.data).copy()
    moved[3] += 1
    ms.points_layer.data = moved
    ms._suggest_anchors()
    assert len(ms.points_layer.data) == 6


def test_no_suggestion_without_selection_or_with_a_half_pair(viewer, scene):
    image_layer, object_list, ms = scene
    with pytest.warns(UserWarning, match="Select an object"):
        ms._suggest_anchors()
    assert len(ms.points_layer.data) == 0
    _select(viewer, object_list)
    ms.points_layer.add(np.array([20., 30., 18.]))
    with pytest.warns(UserWarning, match="Finish the current pair"):
        ms._suggest_anchors()
    assert len(ms.points_layer.data) == 1


def test_button_is_connected_and_follows_the_image_dimension(viewer, scene):
    image_layer, object_list, ms = scene
    _select(viewer, object_list)
    assert ms.suggest_anchors_button.enabled
    ms.suggest_anchors_button.native.click()
    assert len(ms.points_layer.data) == 2
    flat = viewer.add_image(np.zeros((20, 20)), name="flat")
    ms.image_layer = flat
    assert not ms.suggest_anchors_button.enabled
