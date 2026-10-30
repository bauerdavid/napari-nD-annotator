"""The minimal contour overlay can work on a ROI window of the plane and still speaks in plane coordinates."""
import numpy as np
import pytest

from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
from napari_nd_annotator.minimal_contour.plane_features import compute_gradients, to_float64_pair
from napari_nd_annotator.minimal_contour.roi import Box

PLANE = (160, 200)
set_calculator_feature = VispyMinimalContourOverlay.set_calculator_feature


@pytest.fixture(autouse=True)
def calculator():
    """The calculator is shared by the whole process: configure it and leave it without features."""
    VispyMinimalContourOverlay.mc_calculator.set_method(0)
    VispyMinimalContourOverlay.mc_calculator.set_param(5)
    yield VispyMinimalContourOverlay.mc_calculator
    set_calculator_feature(None)
    VispyMinimalContourOverlay.roi_requester = None


@pytest.fixture(scope="module")
def plane():
    """A noisy plane with a bright disc, its three channel image and gradients."""
    yy, xx = np.mgrid[:PLANE[0], :PLANE[1]]
    rng = np.random.default_rng(0)
    values = 200 * ((yy - 80) ** 2 + (xx - 100) ** 2 < 40 ** 2) + rng.random(PLANE) * 20
    image = np.clip(values / 255, 0, 1)
    grad_x, grad_y = to_float64_pair(compute_gradients(values))
    return image, grad_x, grad_y


def _hand_over(plane, box=None):
    """Give the calculator the whole plane, or the window ``box`` of it."""
    image, grad_x, grad_y = plane
    if box is None:
        set_calculator_feature(np.repeat(image[..., None], 3, axis=2), grad_x, grad_y)
        return
    window = box.slices
    set_calculator_feature(np.repeat(image[window][..., None], 3, axis=2), grad_x[window], grad_y[window],
                           roi_origin=box.origin, plane_shape=PLANE)


def _estimate(anchors):
    """What the overlay's mouse move computes for ``anchors`` (rows, columns), without a viewer."""
    overlay = VispyMinimalContourOverlay.__new__(VispyMinimalContourOverlay)
    overlay.point_triangle = np.asarray(anchors, dtype=float)
    return overlay._estimate()


ANCHORS = [[45, 100], [100, 138], [110, 70]]


def test_calculator_state_follows_the_features(plane):
    _hand_over(plane)
    assert VispyMinimalContourOverlay._features_shape == PLANE
    assert VispyMinimalContourOverlay._roi_origin == (0, 0)
    assert VispyMinimalContourOverlay._plane_shape is None

    _hand_over(plane, Box(20, 140, 40, 170))
    assert VispyMinimalContourOverlay._features_shape == (120, 130)
    assert VispyMinimalContourOverlay._roi_origin == (20, 40)
    assert VispyMinimalContourOverlay._plane_shape == PLANE

    _hand_over(plane)  # a later call without a ROI forgets the old one
    assert VispyMinimalContourOverlay._roi_origin == (0, 0)
    assert VispyMinimalContourOverlay._plane_shape is None

    _hand_over(plane, Box(20, 140, 40, 170))
    set_calculator_feature(None)
    assert VispyMinimalContourOverlay._features_shape is None
    assert VispyMinimalContourOverlay._roi_origin == (0, 0)
    assert VispyMinimalContourOverlay._plane_shape is None


def test_contour_on_a_roi_window_equals_the_whole_plane_one(plane):
    _hand_over(plane)
    expected = _estimate(ANCHORS)
    assert len(expected) == 3 and sum(len(s) for s in expected) > 50

    _hand_over(plane, Box(20, 140, 40, 170))
    actual = _estimate(ANCHORS)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)  # in plane coordinates, not window ones


def test_roi_touching_the_plane_edge_needs_no_pad(plane):
    # the plane's own edge behaves the same with or without a window
    anchors = [[10, 20], [60, 90], [20, 150]]
    _hand_over(plane)
    expected = _estimate(anchors)
    _hand_over(plane, Box(0, 120, 0, 200))
    actual = _estimate(anchors)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_search_outside_the_window_asks_for_a_new_window(plane):
    requests = []

    def requester(search_box):
        requests.append(search_box)
        _hand_over(plane)  # the widget answers by giving the calculator the whole plane

    VispyMinimalContourOverlay.roi_requester = requester
    _hand_over(plane)
    expected = _estimate(ANCHORS)

    _hand_over(plane, Box(20, 140, 40, 170))
    outside = [[45, 100], [100, 138], [150, 70]]  # the search window reaches row 150 + margin: outside the window
    _hand_over(plane)
    expected = _estimate(outside)
    _hand_over(plane, Box(20, 140, 40, 170))
    actual = _estimate(outside)
    assert len(requests) == 1 and requests[0].y1 > 140
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)

    # inside the window again: nothing is requested
    _hand_over(plane, Box(20, 140, 40, 170))
    _estimate(ANCHORS)
    assert len(requests) == 1


def test_search_outside_the_window_without_an_answer_returns_none(plane):
    _hand_over(plane, Box(20, 140, 40, 170))
    assert VispyMinimalContourOverlay.roi_requester is None
    assert _estimate([[45, 100], [100, 138], [150, 70]]) is None
    requests = []
    VispyMinimalContourOverlay.roi_requester = requests.append  # a requester that does not change anything
    assert _estimate([[45, 100], [100, 138], [150, 70]]) is None
    assert len(requests) == 1
