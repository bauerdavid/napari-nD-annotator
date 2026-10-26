"""Region of interest (ROI) of the displayed plane that is handed to the minimal contour calculator.

``calculator.set_image`` and the image preparation cost time proportional to the plane area, while a contour is only
ever searched near its anchors. On big planes the calculator therefore gets a window of the plane instead of the whole
plane. This module only does the geometry (no Qt or napari dependency): boxes are ``(y0, y1, x0, x1)`` in the
coordinates of the displayed plane (rows, then columns), end exclusive and clipped to the plane.
"""
import math
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

# an object's box gets 20 % of its size + 32 pixels on every side
MARGIN_FRACTION = 0.2
MARGIN_PIXELS = 32
# without a selected object, only planes with more pixels than this are windowed (2048 x 2048 is 4 M pixels)
LARGE_PLANE_PIXELS = 4 * 1024 * 1024
# the anchor driven window is at least this many pixels along both axes
WINDOW_SIZE = 1024
# a ROI covering more than this share of the plane is not worth it: the whole plane is used
MAX_AREA_FRACTION = 0.6
# the calculator treats the outer pixel ring of its image specially and reads one pixel beyond the search window, so
# the ROI has to reach this far around a search window (unless the ROI side is the edge of the plane)
COVER_PAD = 2


class Box(NamedTuple):
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0

    @property
    def origin(self) -> Tuple[int, int]:
        return self.y0, self.x0

    @property
    def area(self) -> int:
        return max(self.y1 - self.y0, 0) * max(self.x1 - self.x0, 0)

    @property
    def slices(self) -> Tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


def clip(box: Box, plane_shape: Sequence[int]) -> Box:
    """``box`` restricted to a plane of ``plane_shape`` (an empty box stays empty, with y1 >= y0 and x1 >= x0)."""
    height, width = plane_shape[:2]
    y0 = min(max(int(box.y0), 0), height)
    x0 = min(max(int(box.x0), 0), width)
    return Box(y0, min(max(int(box.y1), y0), height), x0, min(max(int(box.x1), x0), width))


def expand(box: Box, plane_shape: Sequence[int], fraction: float = MARGIN_FRACTION,
           pixels: int = MARGIN_PIXELS) -> Box:
    """``box`` grown on every side by ``fraction`` of its size along that axis plus ``pixels``, clipped to the plane."""
    margin_y = int(math.ceil(fraction * (box.y1 - box.y0))) + pixels
    margin_x = int(math.ceil(fraction * (box.x1 - box.x0))) + pixels
    return clip(Box(box.y0 - margin_y, box.y1 + margin_y, box.x0 - margin_x, box.x1 + margin_x), plane_shape)


def union(a: Box, b: Box) -> Box:
    return Box(min(a.y0, b.y0), max(a.y1, b.y1), min(a.x0, b.x0), max(a.x1, b.x1))


def object_box(bbox_min, bbox_max, dims_displayed: Sequence[int], plane_shape: Sequence[int]) -> Optional[Box]:
    """The ROI of an object in the displayed plane, or None when it cannot be derived.

    ``bbox_min`` and ``bbox_max`` are the inclusive corners of the object's bounding box in data coordinates (all
    dimensions of the image). Only the displayed dimensions matter: the object is annotated on any slice, also on
    slices outside of its current extent. ``dims_displayed`` gives the data dimension shown along the rows and the
    columns of the plane, so transposed display orders are handled. The result carries the margin.
    """
    if len(dims_displayed) != 2:
        return None
    try:
        row_dim, col_dim = (int(d) for d in dims_displayed)
        box = Box(int(bbox_min[row_dim]), int(bbox_max[row_dim]) + 1, int(bbox_min[col_dim]), int(bbox_max[col_dim]) + 1)
    except (IndexError, TypeError, ValueError):
        return None
    height, width = plane_shape[:2]
    if box.y1 <= box.y0 or box.x1 <= box.x0 or box.y1 <= 0 or box.x1 <= 0 or box.y0 >= height or box.x0 >= width:
        return None  # empty or outside of the plane
    return expand(box, plane_shape)


def window_around(points, plane_shape: Sequence[int], size: int = WINDOW_SIZE) -> Box:
    """A window of at least ``size`` x ``size`` pixels (as far as the plane allows) around ``points`` (rows, columns).

    It is centred on the bounding box of the points and always contains that box with the usual margin, so a
    contour drawn between the points can be searched without leaving the window. Near the edge of the plane the
    window is moved inwards, not shrunk.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
    height, width = plane_shape[:2]
    low = np.floor(points.min(axis=0)).astype(int)
    high = np.ceil(points.max(axis=0)).astype(int) + 1
    needed = expand(Box(low[0], high[0], low[1], high[1]), plane_shape)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2 + 0.5
    limits = []
    for axis, extent in enumerate((height, width)):
        length = min(max(size, needed[2 * axis + 1] - needed[2 * axis]), extent)
        start = int(round(centre[axis] - length / 2))
        start = min(max(start, 0), extent - length)
        limits += [start, start + length]
    return union(Box(*limits), needed)


def covers(box: Box, search_box: Box, plane_shape: Sequence[int], pad: int = COVER_PAD) -> bool:
    """Whether the calculator image ``box`` lets a search in ``search_box`` give the same result as the full plane.

    The search window must be ``pad`` pixels inside the ROI on every side, except on sides where the ROI ends at
    the edge of the plane (the plane's own edge behaves the same with or without a ROI).
    """
    height, width = plane_shape[:2]
    return (
        (box.y0 <= 0 or box.y0 <= search_box.y0 - pad)
        and (box.y1 >= height or box.y1 >= search_box.y1 + pad)
        and (box.x0 <= 0 or box.x0 <= search_box.x0 - pad)
        and (box.x1 >= width or box.x1 >= search_box.x1 + pad)
    )


def use_roi(box: Optional[Box], plane_shape: Sequence[int], max_fraction: float = MAX_AREA_FRACTION) -> Optional[Box]:
    """``box`` if working on it is worth it (it is clearly smaller than the plane), else None (use the whole plane)."""
    if box is None:
        return None
    box = clip(box, plane_shape)
    height, width = plane_shape[:2]
    if box.area == 0 or box.area > max_fraction * height * width:
        return None
    return box


def feature_padding(rgb: bool, sigma: Optional[float]) -> int:
    """Pixels around a ROI that the features have to be computed on for the result inside it to equal the full plane's.

    The gradients need 1 pixel (Sobel) or 8 (the sigma 2 gaussian derivative of RGB images); blurring the image
    and the gradients afterwards (skimage ``gaussian`` with ``truncate=2``, a radius of 2 sigma) needs that many more.
    """
    padding = 8 if rgb else 1
    if sigma is not None:
        padding += int(math.ceil(2 * sigma)) + 1
    return padding


def padded(box: Box, plane_shape: Sequence[int], padding: int) -> Box:
    """``box`` grown by ``padding`` pixels on every side and clipped to the plane."""
    return clip(Box(box.y0 - padding, box.y1 + padding, box.x0 - padding, box.x1 + padding), plane_shape)
