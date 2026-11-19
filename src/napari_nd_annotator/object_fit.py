"""Fit a bounding box to the object under it, also along the dimensions that the box does not span.

A box drawn on one slice of a 3D image has no extent along the dimension that is not displayed (and the user does not
know where the middle of the object is, or how large it is when drawing from its top). :func:`fit_box_to_object` grows
or shrinks such a box to the bright (or dark) object that it touches, in every dimension.
"""
from typing import NamedTuple, Optional

import numpy as np
from scipy import ndimage as ndi

POLARITIES = ("auto", "bright", "dark")
MIN_CONTRAST = .25  # of the range between the background and the brightest part of the surroundings
MIN_SNR = 4.  # the box content has to be this many noise standard deviations above the background


class BoxFit(NamedTuple):
    """The inclusive corners of the fitted box in data (index) coordinates; ``found`` is False (and ``reason`` says why)
    if there was no object under the box."""
    found: bool = False
    box_min: Optional[np.ndarray] = None
    box_max: Optional[np.ndarray] = None
    polarity: str = ""
    reaches_limit: bool = False  # the object is cut by the search region: it is probably larger (or merged with a neighbor)
    reason: str = ""


def fit_box_to_object(image, box_min, box_max, scale=None, polarity="auto", threshold=.5, sigma=1.,
                      search_factor=2.):
    """The bounding box of the object that the box ``box_min``..``box_max`` (inclusive indices) touches.

    The object is the connected region around the box that is brighter (``polarity="bright"``) or darker (``"dark"``)
    than a threshold half way (``threshold``, 0..1) between the typical value inside the box and the background of the
    surroundings; ``"auto"`` picks whichever the content of the box is closer to. The search region is the box grown by
    ``search_factor`` times its largest size in every direction (a box that is flat in a dimension counts as one voxel
    there), so the box may be rough: too small, off the middle, or drawn on the top slice of the object.

    ``scale`` is the size of a voxel along each axis (e.g. ``(3, 1, 1)`` for a z step three times the pixel size), so
    the smoothing and the search region are isotropic in physical units.

    If the object reaches the border of the search region, or the region holds no background at all (a small box deep
    inside a large object), the region is widened (up to 8 times ``search_factor``) and the search repeated.
    ``BoxFit.reaches_limit`` is set if even the widest region cuts the object.
    """
    image_shape = np.asarray(image.shape)
    ndim = len(image_shape)
    corner_a, corner_b = np.asarray(box_min, dtype=float), np.asarray(box_max, dtype=float)
    box_min, box_max = np.minimum(corner_a, corner_b), np.maximum(corner_a, corner_b)
    if len(box_min) != ndim:
        return BoxFit(reason="the box and the image have different dimensions")
    if polarity not in POLARITIES:
        raise ValueError("polarity must be one of %s, not %r" % (POLARITIES, polarity))
    scale = np.ones(ndim) if scale is None else np.asarray(scale, dtype=float)
    lo = np.clip(np.floor(box_min), 0, image_shape - 1).astype(int)
    hi = np.clip(np.ceil(box_max), 0, image_shape - 1).astype(int)  # inclusive
    if np.any(box_max < 0) or np.any(box_min > image_shape - 1):
        return BoxFit(reason="the box is outside of the image")
    fit = BoxFit()
    for factor in (1, 2, 4, 8):
        fit = _fit_in_region(image, lo, hi, scale, polarity, threshold, sigma, search_factor * factor)
        if fit.found and not fit.reaches_limit:
            break
        if not fit.found and fit.reason not in WIDEN_FOR:
            break  # a wider region does not help
    return fit


ONLY_BACKGROUND = "no object under the box (it holds only background)"
NO_CONTRAST = "the surroundings of the box have no contrast"
WIDEN_FOR = (ONLY_BACKGROUND, NO_CONTRAST)  # failures that a wider search region may fix (it may contain the edge)


def _fit_in_region(image, lo, hi, scale, polarity, threshold, sigma, search_factor):
    image_shape = np.asarray(image.shape)
    size = (hi - lo + 1) * scale  # physical size of the box
    # twice as far where the box is flat: nothing is known there about the extent of the object
    margin_size = np.where(hi == lo, 2., 1.) * search_factor * size.max()
    margin = np.ceil(margin_size / scale).astype(int)
    crop_lo = np.maximum(lo - margin, 0)
    crop_hi = np.minimum(hi + margin + 1, image_shape)  # exclusive
    crop = np.asarray(image[tuple(slice(a, b) for a, b in zip(crop_lo, crop_hi))], dtype=np.float32)
    smooth = ndi.gaussian_filter(crop, sigma * scale.min() / scale, mode="nearest")
    seed = tuple(slice(a - c, b - c + 1) for a, b, c in zip(lo, hi, crop_lo))

    low, high = np.percentile(smooth, (10, 90))
    if high <= low:
        return BoxFit(reason=NO_CONTRAST)
    inside = smooth[seed]
    if polarity == "auto":
        # the box may hold mostly background (e.g. drawn on the top slice of the object, where it is small): decide by
        # which extreme of the box content is farther from the typical value of the surroundings
        typical = np.median(smooth)
        up, down = np.percentile(inside, 90) - typical, typical - np.percentile(inside, 10)
        polarity = "bright" if up >= down else "dark"
    if polarity == "dark":
        smooth, inside, low, high = -smooth, -inside, -high, -low
    foreground = np.percentile(inside, 90)
    cut = low + threshold * (foreground - low)
    if foreground <= low:
        return BoxFit(polarity=polarity, reason="the box does not contain anything brighter than its surroundings"
                      if polarity == "bright" else "the box does not contain anything darker than its surroundings")
    # the box must stand out of the noise of the background, and reach a good part of the brightest of its surroundings,
    # otherwise it is on the background
    background = smooth[smooth <= np.median(smooth)]
    noise = 1.4826 * np.median(np.abs(background - np.median(smooth)))
    if foreground - low < MIN_SNR * noise or foreground - low < MIN_CONTRAST * (np.percentile(smooth, 99) - low):
        return BoxFit(polarity=polarity, reason=ONLY_BACKGROUND)
    labels, n = ndi.label(smooth > cut)
    overlap = np.bincount(labels[seed].ravel(), minlength=n + 1)
    overlap[0] = 0
    if n == 0 or overlap.max() == 0:
        return BoxFit(polarity=polarity, reason="no object under the box")
    object_slices = ndi.find_objects((labels == overlap.argmax()).astype(np.uint8))[0]
    new_min = np.array([s.start for s in object_slices]) + crop_lo
    new_max = np.array([s.stop - 1 for s in object_slices]) + crop_lo
    # cut by the search region (but not by the image border, there the object really ends)
    touches_low = (new_min == crop_lo) & (crop_lo > 0)
    touches_high = (new_max == crop_hi - 1) & (crop_hi < image_shape)
    return BoxFit(True, new_min, new_max, polarity, bool(np.any(touches_low | touches_high)))
