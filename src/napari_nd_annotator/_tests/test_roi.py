"""Geometry of the minimal contour ROI (no napari or Qt needed)."""
import numpy as np
import pytest

from napari_nd_annotator.minimal_contour import roi
from napari_nd_annotator.minimal_contour.roi import Box

PLANE = (1000, 1200)


def test_box_properties():
    box = Box(10, 30, 5, 50)
    assert box.shape == (20, 45)
    assert box.origin == (10, 5)
    assert box.area == 900
    data = np.arange(60 * 60).reshape(60, 60)
    assert data[box.slices].shape == (20, 45)


def test_clip_keeps_the_box_inside_the_plane():
    assert roi.clip(Box(-5, 20, -1, 30), PLANE) == Box(0, 20, 0, 30)
    assert roi.clip(Box(990, 1100, 1190, 1300), PLANE) == Box(990, 1000, 1190, 1200)
    empty = roi.clip(Box(2000, 2100, 5, 9), PLANE)
    assert empty.area == 0 and empty.y1 >= empty.y0


def test_expand_adds_a_fraction_and_pixels_on_every_side():
    box = Box(400, 500, 300, 600)
    grown = roi.expand(box, PLANE)
    assert grown == Box(400 - 20 - 32, 500 + 20 + 32, 300 - 60 - 32, 600 + 60 + 32)
    assert roi.expand(Box(0, 10, 0, 10), PLANE).origin == (0, 0)  # clipped at the plane edge
    assert roi.expand(Box(990, 1000, 1190, 1200), PLANE)[1::2] == PLANE


def test_object_box_uses_the_displayed_dimensions_and_inclusive_corners():
    bbox_min, bbox_max = [3, 100, 200], [9, 149, 399]  # (z, y, x), inclusive
    rows_cols = roi.object_box(bbox_min, bbox_max, (1, 2), PLANE)
    assert rows_cols == roi.expand(Box(100, 150, 200, 400), PLANE)
    # transposed display: rows show x, columns show y
    transposed = roi.object_box(bbox_min, bbox_max, (2, 1), PLANE)
    assert transposed == roi.expand(Box(200, 400, 100, 150), PLANE)
    # slice dimensions are not displayed: z does not matter
    assert roi.object_box([0, 100, 200], [50, 149, 399], (1, 2), PLANE) == rows_cols
    # displaying z and x
    assert roi.object_box(bbox_min, bbox_max, (0, 2), (30, 1200)) == roi.expand(Box(3, 10, 200, 400), (30, 1200))


@pytest.mark.parametrize("bbox_min, bbox_max, dims", [
    ([0, 0], [10, 10], (0, 1, 2)),  # 3 displayed dims
    ([0, 0], [10, 10], (0, 2)),  # dimension out of range
    ([2000, 2000], [2100, 2100], (0, 1)),  # outside of the plane
    ([-50, -50], [-10, -10], (0, 1)),
    ([10, 10], [5, 5], (0, 1)),  # empty
    (None, None, (0, 1)),
])
def test_object_box_returns_none_when_it_cannot_be_derived(bbox_min, bbox_max, dims):
    assert roi.object_box(bbox_min, bbox_max, dims, PLANE) is None


def test_object_box_partly_outside_the_plane_is_clipped():
    box = roi.object_box([-20, 900], [30, 1150], (0, 1), PLANE)
    assert box == Box(0, 30 + 1 + 11 + 32, 900 - 51 - 32, 1200)  # 20 % of 51 rows is 11 (rounded up), of 251 columns 51


def test_window_around_points_has_the_minimum_size_and_is_centred():
    box = roi.window_around([[500, 600], [520, 640]], (4000, 4000), size=1024)
    assert box.shape == (1024, 1024)
    cy, cx = (box.y0 + box.y1) / 2, (box.x0 + box.x1) / 2
    assert abs(cy - 511) <= 2 and abs(cx - 621) <= 2


def test_window_around_points_is_moved_inwards_at_the_plane_edge():
    corner = roi.window_around([[5, 3]], (4000, 4000), size=1024)
    assert corner == Box(0, 1024, 0, 1024)
    far = roi.window_around([[3995, 3999]], (4000, 4000), size=1024)
    assert far == Box(2976, 4000, 2976, 4000)


def test_window_around_contains_far_apart_points_with_their_margin():
    points = np.array([[100., 100.], [1900., 2500.]])
    box = roi.window_around(points, (4000, 4000), size=1024)
    needed = roi.expand(Box(100, 1901, 100, 2501), (4000, 4000))
    assert box.y0 <= needed.y0 and box.y1 >= needed.y1 and box.x0 <= needed.x0 and box.x1 >= needed.x1


def test_window_around_on_a_small_plane_is_the_plane():
    assert roi.window_around([[10, 10]], (300, 200), size=1024) == Box(0, 300, 0, 200)


def test_covers_needs_the_pad_except_at_the_plane_edge():
    box = Box(100, 500, 200, 600)
    assert roi.covers(box, Box(110, 490, 210, 590), PLANE)
    assert roi.covers(box, Box(102, 498, 202, 598), PLANE)
    assert not roi.covers(box, Box(101, 490, 210, 590), PLANE)  # inside by one pixel only
    assert not roi.covers(box, Box(110, 499, 210, 590), PLANE)
    assert not roi.covers(box, Box(110, 490, 199, 590), PLANE)
    assert not roi.covers(box, Box(110, 490, 210, 601), PLANE)
    # ROI sides on the plane edge need no pad
    edge = Box(0, 500, 0, 1200)
    assert roi.covers(edge, Box(0, 490, 0, 1200), PLANE)
    assert not roi.covers(edge, Box(0, 499, 0, 1200), PLANE)


def test_use_roi_rejects_big_or_empty_boxes():
    assert roi.use_roi(None, PLANE) is None
    assert roi.use_roi(Box(0, 1000, 0, 1200), PLANE) is None
    assert roi.use_roi(Box(0, 900, 0, 1200), PLANE) is None  # 90 % of the plane
    assert roi.use_roi(Box(5000, 6000, 0, 10), PLANE) is None
    assert roi.use_roi(Box(100, 300, 100, 300), PLANE) == Box(100, 300, 100, 300)
    assert roi.use_roi(Box(-10, 300, 100, 300), PLANE) == Box(0, 300, 100, 300)


@pytest.mark.parametrize("rgb, sigma, expected", [
    (False, None, 1),
    (True, None, 8),
    (False, 1.0, 1 + 2 + 1),
    (False, 3.0, 1 + 6 + 1),
    (True, 2.5, 8 + 5 + 1),
])
def test_feature_padding(rgb, sigma, expected):
    assert roi.feature_padding(rgb, sigma) == expected


def test_padded_is_clipped():
    assert roi.padded(Box(2, 50, 100, 200), PLANE, 8) == Box(0, 58, 92, 208)
    assert roi.padded(Box(0, 1000, 0, 1200), PLANE, 8) == Box(0, 1000, 0, 1200)
