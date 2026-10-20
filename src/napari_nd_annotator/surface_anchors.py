"""Suggest the two surface anchor points of the minimal surface tool from a bounding box.

The minimal surface method needs two points ON the surface of the object that are roughly antipodal (they are used as
the diameter of a sphere, see ``pts_2_bb`` in ``_widgets/minimal_surface_widget.py``). When the object already has a
bounding box (object list), a good first guess for these points can be found automatically: cast a ray from the box
centre along the longest axis in both directions and take the first strong boundary on each side.

Everything here is plain numpy/scipy and independent of napari and Qt. Coordinates are data (index) coordinates, in the
axis order of ``phi``; ``scale`` (voxel size per axis) is only used to measure distances.
"""
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

__all__ = ["AnchorSuggestion", "suggest_surface_anchors", "edge_cost_from_image", "suggest_anchors_in_image"]


@dataclass
class AnchorSuggestion:
    """Result of :func:`suggest_surface_anchors`.

    ``p1``/``p2`` are float arrays in data coordinates (``None`` if there is no suggestion, then ``reason`` says why).
    ``p1`` is on the lower-index side of the box centre along ``axis``.

    Diagnostics (all per point, order ``(p1, p2)``):

    - ``phi_ratio``: phi at the point divided by the median phi in the bounding box (low = strong boundary)
    - ``drop``: relative depth of the boundary, ``(flat - phi) / flat`` of the smoothed phi on the ray, where
      ``flat`` is the 90th percentile of phi in the box
    - ``zscore``: depth of the boundary in units of the robust spread (MAD) of phi in the box

    ``confidence`` is the weaker of the two ``drop`` values (0..1), ``distance`` the physical distance of the points,
    ``axis`` the axis the ray was cast along.
    """
    p1: Optional[np.ndarray] = None
    p2: Optional[np.ndarray] = None
    axis: Optional[int] = None
    distance: float = 0.
    confidence: float = 0.
    phi_ratio: tuple = (np.nan, np.nan)
    drop: tuple = (0., 0.)
    zscore: tuple = (0., 0.)
    reason: str = ""

    @property
    def found(self):
        return self.p1 is not None and self.p2 is not None

    def __bool__(self):
        return self.found


def suggest_surface_anchors(
        bbox_min,
        bbox_max,
        phi,
        scale=1.,
        ray_smoothing=1.,
        snap_radius=3.,
        snap_lateral_radius=1.,
        snap_smoothing=.7,
        first_min_fraction=.9,
        min_drop=0.,
        min_zscore=4.,
        wall_tolerance=2.,
        axis_strategy="fallback",
        method="axis",
        n_rays=256,
        max_radius_jump=1.3,
        min_relative_strength=.8,
):
    """Propose two anchor points on the surface of the object inside a bounding box.

    From the box centre a ray is cast in both directions along an axis (restricted to the box). The profile of the
    smoothed ``phi`` along the ray is searched, from the centre outwards, for the FIRST strong local minimum, i.e. the
    first one whose depth is at least ``first_min_fraction`` of the deepest minimum on that side. This prefers the
    boundary of the object over a deeper edge further out (a neighbouring object, the box wall). The point is then
    snapped to the lowest ``phi`` in a small neighbourhood.

    :param bbox_min: lower corner of the bounding box, data (index) coordinates, one value per axis of ``phi``
    :param bbox_max: upper corner of the bounding box (inclusive)
    :param phi: edge cost with the same axes as the image, LOW on boundaries and HIGH in flat regions, as used by the
        surface method (``alpha + (1-alpha)*exp(-beta*features)``). Only the neighbourhood of the box is read.
    :param scale: voxel size, scalar or one value per axis. Rays are cast along the axis with the largest
        PHYSICAL box extent and all radii are scale aware.
    :param ray_smoothing: sigma (in voxels of the smallest voxel size) of the gaussian smoothing used for the ray
        profile; suppresses noise and texture
    :param snap_radius: radius (same units) of the snapping neighbourhood along the ray direction
    :param snap_lateral_radius: radius of the snapping neighbourhood perpendicular to the ray. Kept small on purpose:
        the boundary is a plateau of low phi, a wide lateral search lets the points drift along the surface and
        spoils the antipodal geometry
    :param snap_smoothing: sigma of the (lighter) smoothing used for snapping
    :param first_min_fraction: a local minimum counts as strong if its depth is at least this fraction of the deepest
        local minimum on the same side of the centre
    :param min_drop: minimal relative depth of the boundary, ``(flat - phi) / flat`` with ``flat`` the 90th percentile of
        phi in the box, otherwise there is no suggestion (disabled by default,
        the z-score is the more robust criterion: real edge costs can be shallow in absolute terms)
    :param min_zscore: minimal depth of the boundary in units of the robust standard deviation (MAD) of ``phi`` in
        the box, otherwise there is no suggestion. This only rejects (numerically) flat phi, noise is not
        separated from weak edges by it; use ``confidence``/``zscore`` of the result to judge a suggestion
    :param wall_tolerance: how many voxels beyond the box wall the ray still looks for a boundary (a tight box has
        the surface right on its wall). The returned points are always inside the box.
    :param axis_strategy: ``"longest"`` casts only along the longest axis, ``"fallback"`` (default) tries the next
        longest axes if there is no valid pair on the longest, ``"best"`` tries all axes and returns the pair with
        the highest ``confidence * distance`` (the distance matters: the sphere built from the points should be as large as
        the object)
    :param method: ``"axis"`` (default) casts rays along the box axes only, see above. ``"fan"`` (3D only) casts
        ``n_rays`` rays from the box centre in near uniformly distributed directions (angles measured in physical
        space), takes the first strong minimum on each as above, drops isolated far points (see
        ``max_radius_jump``) and returns the FARTHEST pair of the resulting surface points: a diameter of the object
        that does not have to follow the box axes. Snapping is not applied, the minimum on each ray is already
        located with sub-voxel precision. ``axis_strategy`` is ignored; ``axis`` of the result is the box axis the
        two points are the most aligned with.
    :param n_rays: number of rays of the ``"fan"`` method
    :param max_radius_jump: ``"fan"`` only: a ray whose boundary is farther from the centre than this factor times
        the median distance of its nearest neighbouring rays is considered a leak (neighbouring object, spurious
        far edge) and ignored. The factor has to leave room for the tips of elongated objects.
    :param min_relative_strength: ``"fan"`` only: rays whose boundary depth (``drop``) is below this fraction of the
        median over all rays are ignored
    :return: :class:`AnchorSuggestion`; check ``.found`` (or truthiness) before using the points
    """
    if axis_strategy not in ("longest", "fallback", "best"):
        raise ValueError("axis_strategy must be 'longest', 'fallback' or 'best'")
    if method not in ("axis", "fan"):
        raise ValueError("method must be 'axis' or 'fan'")
    if method == "fan" and phi.ndim != 3:
        raise ValueError("the fan method needs a 3D phi")
    ndim = phi.ndim
    bbox_min = np.asarray(bbox_min, dtype=float).reshape(-1)
    bbox_max = np.asarray(bbox_max, dtype=float).reshape(-1)
    scale = np.broadcast_to(np.asarray(scale, dtype=float), (ndim,))
    if bbox_min.shape != (ndim,) or bbox_max.shape != (ndim,):
        raise ValueError("bbox corners must have one coordinate per axis of phi")
    bbox_min, bbox_max = np.minimum(bbox_min, bbox_max), np.maximum(bbox_min, bbox_max)
    # voxels that can be returned: inside both the box and the array
    lo_idx = np.maximum(np.ceil(bbox_min), 0).astype(int)
    hi_idx = np.minimum(np.floor(bbox_max), np.asarray(phi.shape) - 1).astype(int)
    if np.any(lo_idx > hi_idx):
        return AnchorSuggestion(reason="bounding box does not contain any voxel")

    smallest = scale.min()
    pad = int(np.ceil(wall_tolerance + 3 * ray_smoothing + snap_radius + snap_lateral_radius)) + 1
    pad_per_axis = np.ceil(pad * smallest / scale).astype(int)
    crop_lo = np.maximum(lo_idx - pad_per_axis, 0)
    crop_hi = np.minimum(hi_idx + pad_per_axis + 1, phi.shape)
    crop = tuple(slice(a, b) for a, b in zip(crop_lo, crop_hi))
    phi_crop = np.asarray(phi[crop], dtype=float)
    if not np.all(np.isfinite(phi_crop)):
        return AnchorSuggestion(reason="phi contains non-finite values")
    # gaussian filters that are isotropic in physical space
    ray_phi = gaussian_filter(phi_crop, ray_smoothing * smallest / scale, mode="nearest")
    snap_phi = gaussian_filter(phi_crop, snap_smoothing * smallest / scale, mode="nearest")

    inside = tuple(slice(a - o, b - o + 1) for a, b, o in zip(lo_idx, hi_idx, crop_lo))
    box_values = ray_phi[inside]
    median = float(np.median(box_values))
    # Level of the flat regions. The median is not usable for this: a tight box around a small object is mostly
    # boundary after smoothing, so an upper percentile is taken instead.
    reference = float(np.percentile(box_values, 90))
    if reference <= 0:
        return AnchorSuggestion(reason="phi in the box is not positive")
    # robust spread of the flat level (MAD of the upper quartile); the floor keeps the z-score finite for (almost)
    # constant phi
    top = box_values[box_values >= np.percentile(box_values, 75)]
    spread = max(1.4826 * float(np.median(np.abs(top - np.median(top)))), 1e-3 * reference, 1e-12)

    centre = (bbox_min + bbox_max) / 2
    if method == "fan":
        return _suggest_fan(
            centre, bbox_min, bbox_max, lo_idx, hi_idx, crop_lo, scale, ray_phi, snap_phi, reference, median, spread,
            first_min_fraction, min_drop, min_zscore, wall_tolerance, n_rays, max_radius_jump,
            min_relative_strength
        )
    extents = (bbox_max - bbox_min) * scale
    axes = [int(a) for a in np.argsort(-extents, kind="stable")]
    if axis_strategy == "longest":
        axes = axes[:1]
    best = None
    reasons = []
    for axis in axes:
        candidate = _suggest_along_axis(
            axis, centre, bbox_min, bbox_max, lo_idx, hi_idx, crop_lo, scale, ray_phi, snap_phi, reference, median,
            spread, snap_radius, snap_lateral_radius, first_min_fraction, min_drop, min_zscore, wall_tolerance
        )
        if not candidate.found:
            reasons.append("axis %d: %s" % (axis, candidate.reason))
            continue
        if axis_strategy != "best":
            return candidate
        if best is None or _pair_score(candidate) > _pair_score(best):
            best = candidate
    if best is not None:
        return best
    return AnchorSuggestion(reason="no boundary found (%s)" % "; ".join(reasons))


def edge_cost_from_image(image, alpha=.01, beta=5., blur_func=None):
    """Edge cost (phi) of an image like the gradient feature of the minimal surface widget does it.

    The image is normalized to 0..1, optionally blurred by ``blur_func``, the gradient magnitude (central
    differences, in voxel units) ``f`` gives ``phi = alpha' + (1 - alpha') * exp(-beta * f)`` with
    ``alpha' = (alpha - exp(-beta)) / (1 - exp(-beta))`` and phi is divided by its maximum. Low values are boundaries.
    """
    image = np.asarray(image, dtype=float)
    value_range = image.max() - image.min()
    if value_range > 0:
        image = (image - image.min()) / value_range
    else:
        image = np.zeros_like(image)
    if blur_func is not None:
        image = np.asarray(blur_func(image), dtype=float)
    gradient = np.sqrt(sum(g ** 2 for g in np.gradient(image)))
    alpha = (alpha - np.exp(-beta)) / (1 - np.exp(-beta))
    phi = alpha + (1 - alpha) * np.exp(-beta * gradient)
    return phi / phi.max()


def suggest_anchors_in_image(image, bbox_min, bbox_max, scale=1., alpha=.01, beta=5., blur_func=None, margin=10,
                             resample=None, **kwargs):
    """:func:`suggest_surface_anchors` for a raw image: phi is computed only on the bounding box plus ``margin``.

    ``bbox_min``/``bbox_max`` are in the data (index) coordinates of ``image``, so are the returned points.
    The extra keyword arguments are passed on to :func:`suggest_surface_anchors`. As in the minimal surface run, the
    image of the crop is normalized on its own. ``blur_func`` is applied to the normalized crop before the gradient
    is taken, so the anchors are found on the same blurred image the surface is computed on.

    ``resample`` (one factor per axis, e.g. ``(z_scale, 1, 1)``) resamples the crop with a cubic spline first, as the
    minimal surface run does for anisotropic images; the blur and the edge cost then act on isotropic voxels and
    ``scale`` is ignored. Use either ``scale`` or ``resample``.
    """
    shape = np.asarray(image.shape)
    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)
    crop_lo = np.maximum(np.floor(np.minimum(bbox_min, bbox_max)) - margin, 0).astype(int)
    crop_hi = np.minimum(np.ceil(np.maximum(bbox_min, bbox_max)) + margin + 1, shape).astype(int)
    if np.any(crop_lo >= crop_hi):
        return AnchorSuggestion(reason="bounding box is outside of the image")
    crop = np.asarray(image[tuple(slice(a, b) for a, b in zip(crop_lo, crop_hi))])
    # crop index -> index of the (resampled) crop that the edge cost is computed on
    factor = np.ones(len(shape))
    if resample is not None and not np.allclose(resample, 1):
        crop = zoom(crop.astype(float), resample)
        # zoom maps the first and last voxel centres onto each other
        factor = (np.asarray(crop.shape) - 1) / np.maximum(crop_hi - crop_lo - 1, 1)
        scale = 1.
    phi = edge_cost_from_image(crop, alpha, beta, blur_func)
    result = suggest_surface_anchors((bbox_min - crop_lo) * factor, (bbox_max - crop_lo) * factor, phi, scale,
                                     **kwargs)
    if result.found:
        result.p1 = result.p1 / factor + crop_lo
        result.p2 = result.p2 / factor + crop_lo
    return result


def _pair_score(suggestion):
    return suggestion.confidence * suggestion.distance


def _suggest_along_axis(axis, centre, bbox_min, bbox_max, lo_idx, hi_idx, crop_lo, scale, ray_phi, snap_phi,
                        reference, median, spread, snap_radius, snap_lateral_radius, first_min_fraction, min_drop,
                        min_zscore, wall_tolerance):
    n_steps = int(np.floor((bbox_max[axis] - bbox_min[axis]) / 2 + wall_tolerance))
    if n_steps < 2:
        return AnchorSuggestion(axis=axis, reason="box too small along the axis")
    steps = np.arange(n_steps + 1, dtype=float)
    points = []
    drops = []
    zscores = []
    for sign in (-1, 1):
        coords = np.tile(centre - crop_lo, (n_steps + 1, 1))
        coords[:, axis] += sign * steps
        profile = map_coordinates(ray_phi, coords.T, order=1, mode="nearest")
        step = _first_strong_minimum(profile, reference, first_min_fraction)
        if step is None:
            return AnchorSuggestion(axis=axis, reason="no local minimum of phi along the ray")
        depth = reference - profile[int(round(step))]
        drop = depth / reference
        zscore = depth / spread
        if drop < min_drop or zscore < min_zscore:
            return AnchorSuggestion(
                axis=axis, reason="boundary too weak (drop %.2f, z-score %.1f)" % (drop, zscore)
            )
        point = centre.copy()
        point[axis] += sign * step
        points.append(_snap(point, axis, lo_idx, hi_idx, crop_lo, scale, snap_phi, snap_radius,
                            snap_lateral_radius))
        drops.append(float(drop))
        zscores.append(float(zscore))
    p1, p2 = points
    if np.all(p1 == p2):
        return AnchorSuggestion(axis=axis, reason="both anchors collapsed to the same voxel")
    phi_ratio = tuple(
        float(snap_phi[tuple(np.round(p).astype(int) - crop_lo)] / median) for p in points
    )
    return AnchorSuggestion(
        p1=p1,
        p2=p2,
        axis=axis,
        distance=float(np.sqrt((((p1 - p2) * scale) ** 2).sum())),
        confidence=float(np.clip(min(drops), 0, 1)),
        phi_ratio=phi_ratio,
        drop=tuple(drops),
        zscore=tuple(zscores),
    )


def fibonacci_directions(n):
    """``n`` near uniformly distributed unit vectors on the sphere, shape ``(n, 3)``."""
    i = np.arange(n) + .5
    z = 1 - 2 * i / n
    azimuth = np.pi * (1 + 5 ** .5) * i
    r = np.sqrt(1 - z ** 2)
    return np.stack([z, r * np.sin(azimuth), r * np.cos(azimuth)], axis=1)


def _suggest_fan(centre, bbox_min, bbox_max, lo_idx, hi_idx, crop_lo, scale, ray_phi, snap_phi, reference, median,
                 spread, first_min_fraction, min_drop, min_zscore, wall_tolerance, n_rays, max_radius_jump,
                 min_relative_strength):
    smallest = scale.min()
    directions = fibonacci_directions(n_rays)
    # distance (physical) at which each ray leaves the box, extended by the wall tolerance
    half_extent = (bbox_max - bbox_min) / 2 * scale + wall_tolerance * smallest
    with np.errstate(divide="ignore"):
        t_max = np.min(half_extent / np.abs(directions), axis=1)
    n_steps = int(np.floor(t_max / smallest).max())
    steps = np.arange(n_steps + 1) * smallest
    # sample positions (index space) of all rays at once: shape (n_rays, n_steps + 1, 3)
    positions = centre + steps[None, :, None] * (directions / scale)[:, None, :]
    profiles = map_coordinates(
        ray_phi, (positions - crop_lo).reshape(-1, 3).T, order=1, mode="nearest"
    ).reshape(n_rays, n_steps + 1)
    radii = np.full(n_rays, np.nan)
    drops = np.zeros(n_rays)
    zscores = np.zeros(n_rays)
    for i in range(n_rays):
        length = int(t_max[i] // smallest) + 1
        if length < 3:
            continue
        profile = profiles[i, :length]
        step = _first_strong_minimum(profile, reference, first_min_fraction)
        if step is None:
            continue
        depth = reference - profile[int(round(step))]
        drops[i] = depth / reference
        zscores[i] = depth / spread
        if drops[i] >= min_drop and zscores[i] >= min_zscore:
            radii[i] = step * smallest
    valid = np.isfinite(radii)
    if valid.sum() >= 2:
        # the farthest pair favours false far boundaries (extreme value of noise): rays whose boundary is much
        # weaker than the typical one of this object are not trusted
        valid &= drops >= min_relative_strength * np.median(drops[valid])
    if valid.sum() < 2:
        return AnchorSuggestion(reason="no boundary found along the rays")
    # leaks: boundaries much farther away than those of the neighbouring directions
    n_neighbours = min(8, n_rays - 1)
    neighbours = np.argsort(-directions @ directions.T, axis=1)[:, 1:n_neighbours + 1]
    neighbour_radii = np.where(valid[neighbours], radii[neighbours], np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        neighbour_median = np.nanmedian(neighbour_radii, axis=1)
    keep = valid & ~(radii > max_radius_jump * neighbour_median)
    if keep.sum() < 2:
        return AnchorSuggestion(reason="no consistent boundary found along the rays")
    indices = np.flatnonzero(keep)
    points_physical = directions[indices] * radii[indices, None]
    # the farthest pair; a pair of points is at most 2 * the largest radius apart
    distances = np.sqrt(((points_physical[:, None, :] - points_physical[None, :, :]) ** 2).sum(-1))
    a, b = np.unravel_index(np.argmax(distances), distances.shape)
    picked = indices[[a, b]]
    low, high = lo_idx.astype(float), hi_idx.astype(float)
    points = [np.clip(centre + directions[i] / scale * radii[i], low, high) for i in picked]
    direction = (points[1] - points[0]) * scale
    axis = int(np.argmax(np.abs(direction)))
    if points[0][axis] > points[1][axis]:
        points.reverse()
        picked = picked[::-1]
    p1, p2 = points
    if np.all(p1 == p2):
        return AnchorSuggestion(axis=axis, reason="both anchors collapsed to the same point")
    phi_ratio = tuple(
        float(map_coordinates(snap_phi, (p - crop_lo)[:, None], order=1, mode="nearest")[0] / median)
        for p in points
    )
    return AnchorSuggestion(
        p1=p1,
        p2=p2,
        axis=axis,
        distance=float(np.sqrt((((p1 - p2) * scale) ** 2).sum())),
        confidence=float(np.clip(min(drops[picked]), 0, 1)),
        phi_ratio=phi_ratio,
        drop=tuple(float(d) for d in drops[picked]),
        zscore=tuple(float(z) for z in zscores[picked]),
    )


def _first_strong_minimum(profile, reference, first_min_fraction):
    """Sub-sample position of the first local minimum (from index 0) that is deep enough, or None."""
    depth = reference - profile
    is_min = np.zeros(profile.shape, dtype=bool)
    # plateau-safe: lower or equal on the left, strictly lower on the right; the last sample counts as a
    # minimum if the profile is still decreasing towards it
    is_min[1:-1] = (profile[1:-1] <= profile[:-2]) & (profile[1:-1] < profile[2:])
    is_min[-1] = profile[-1] < profile[-2]
    candidates = np.flatnonzero(is_min)
    if candidates.size == 0:
        return None
    threshold = first_min_fraction * depth[candidates].max()
    k = int(candidates[np.argmax(depth[candidates] >= threshold)])
    if 0 < k < len(profile) - 1:
        # parabolic interpolation through the three samples around the minimum
        denominator = profile[k - 1] - 2 * profile[k] + profile[k + 1]
        if denominator > 0:
            return k + float(np.clip(.5 * (profile[k - 1] - profile[k + 1]) / denominator, -.5, .5))
    return float(k)


def _snap(point, axis, lo_idx, hi_idx, crop_lo, scale, snap_phi, snap_radius, snap_lateral_radius):
    """Voxel with the lowest phi in an ellipsoidal neighbourhood (long along ``axis``) of ``point``, inside the box."""
    smallest = scale.min()
    radii = np.full(len(point), snap_lateral_radius * smallest)
    radii[axis] = snap_radius * smallest
    half = np.ceil(radii / scale).astype(int)
    centre_idx = np.clip(np.round(point).astype(int), lo_idx, hi_idx)
    start = np.maximum(centre_idx - half, lo_idx)
    stop = np.minimum(centre_idx + half, hi_idx) + 1
    grids = np.meshgrid(*[np.arange(a, b) for a, b in zip(start, stop)], indexing="ij")
    normalized_sq = sum((((g - p) * s) / r) ** 2 for g, p, s, r in zip(grids, point, scale, radii))
    values = snap_phi[tuple(slice(a - o, b - o) for a, b, o in zip(start, stop, crop_lo))]
    # a vanishing penalty prefers the closest voxel among (almost) equal ones
    cost = np.where(normalized_sq <= 1, values + 1e-6 * normalized_sq, np.inf)
    if not np.isfinite(cost).any():
        return centre_idx.astype(float)
    best = np.unravel_index(np.argmin(cost), cost.shape)
    return (start + np.asarray(best)).astype(float)
