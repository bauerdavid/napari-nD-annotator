"""Surface anchors of the objects of the object list, and the placement of anchors with the cursor ray.

The minimal surface tool needs two points on the surface of an object (see ``surface_anchors.py``). Here they belong
to the object, an entry of the object list identified by its label id. Everything is plain numpy/scipy and independent
of napari and Qt; coordinates are data (index) coordinates of the image layer of the object list.

- :class:`AnchorStore`: the anchors of each object, kept in a dict (``bbox_layer.metadata``, see :func:`anchor_store`)
  so that they follow the bounding box layer and survive the selection of other objects.
- :func:`pick_along_ray`: the point where the cursor ray first enters the data inside a bounding box.
- :func:`pick_on_plane`: the point where the cursor ray crosses a plane inside a bounding box.
- :func:`nearest_point_to_ray`: the anchor under the cursor, to grab it.
"""
import weakref
from typing import Optional

import numpy as np
from scipy.ndimage import map_coordinates

__all__ = ["METADATA_KEY", "AnchorStore", "anchor_store", "clip_ray_to_box", "box_reference", "pick_along_ray",
           "pick_on_plane", "nearest_point_to_ray"]

METADATA_KEY = "nd_annotator_anchors"
_STORES = weakref.WeakKeyDictionary()


class AnchorStore:
    """The anchor points (at most two) of each object, by label id.

    The points live in ``storage``, a dict ``{str(label): [[...], [...]]}`` of plain lists, so that it can be saved as
    JSON. A pair is complete with two points; adding a point to a complete pair starts a new one.
    """

    def __init__(self, storage=None):
        self.storage = {} if storage is None else storage

    def get(self, label) -> np.ndarray:
        """The anchors of the object as an ``(n, ndim)`` array, n being 0, 1 or 2."""
        points = self.storage.get(str(int(label)), [])
        return np.asarray(points, dtype=float).reshape(len(points), -1) if points else np.zeros((0, 3))

    def has_pair(self, label) -> bool:
        return len(self.storage.get(str(int(label)), [])) == 2

    def set(self, label, points):
        points = [[float(c) for c in point] for point in points]
        if len(points) > 2:
            raise ValueError("an object has at most two anchors")
        if points:
            self.storage[str(int(label))] = points
        else:
            self.storage.pop(str(int(label)), None)

    def add(self, label, point) -> np.ndarray:
        """Adds a point; the first of a new pair if the object already has two. Returns the anchors."""
        points = self.get(label)
        point = np.asarray(point, dtype=float)
        self.set(label, [point] if len(points) >= 2 else [*points, point])
        return self.get(label)

    def pop_last(self, label):
        points = self.get(label)
        if len(points):
            self.set(label, points[:-1])

    def clear(self, label):
        if str(int(label)) in self.storage:
            self.set(label, [])


def anchor_store(bbox_layer) -> AnchorStore:
    """The anchor store of a bounding box layer. It is shared by everything that asks for it."""
    storage = bbox_layer.metadata.setdefault(METADATA_KEY, {})
    store = _STORES.get(bbox_layer)
    if store is None or store.storage is not storage:  # the metadata can be replaced
        store = _STORES[bbox_layer] = AnchorStore(storage)
    return store


def clip_ray_to_box(start, end, box_min, box_max) -> Optional[tuple]:
    """The part of the segment ``start + t * (end - start)`` inside the box, as ``(t_enter, t_exit)`` in 0..1.

    The box is given by the voxel indices of its corners (both inclusive), so it extends half a voxel further.
    Returns None if the segment misses the box.
    """
    start = np.asarray(start, dtype=float)
    direction = np.asarray(end, dtype=float) - start
    lower = np.asarray(box_min, dtype=float) - .5
    upper = np.asarray(box_max, dtype=float) + .5
    t_enter, t_exit = 0., 1.
    for s, d, lo, hi in zip(start, direction, lower, upper):
        if abs(d) < 1e-12:
            if s < lo or s > hi:
                return None
            continue
        t0, t1 = sorted(((lo - s) / d, (hi - s) / d))
        t_enter, t_exit = max(t_enter, t0), min(t_exit, t1)
        if t_enter > t_exit:
            return None
    return t_enter, t_exit


def box_reference(volume, box_min, box_max) -> tuple:
    """``(background, peak)``: the 1st and 99th percentile of the volume inside the box (inclusive voxel corners).

    Computed on a strided subsample, so it is cheap for big boxes. :func:`pick_along_ray` takes it as ``reference``
    to avoid recomputing it for every position of a dragged anchor.
    """
    shape = np.asarray(volume.shape)
    corner_low = np.clip(np.ceil(np.asarray(box_min, dtype=float)).astype(int), 0, shape - 1)
    corner_high = np.clip(np.floor(np.asarray(box_max, dtype=float)).astype(int) + 1, 1, shape)
    box = volume[tuple(slice(a, b) for a, b in zip(corner_low, corner_high))]
    stride = max(int(np.ceil((box.size / 1e6) ** (1 / box.ndim))), 1)
    background, peak = np.percentile(np.asarray(box[(slice(None, None, stride),) * box.ndim]), (1, 99))
    return float(background), float(peak)


def pick_along_ray(volume, start, end, box_min, box_max, fraction=.5, min_contrast=.2, step=.5,
                   reference=None) -> Optional[np.ndarray]:
    """Where a ray first enters the data inside a bounding box.

    The ray from ``start`` to ``end`` (data coordinates, e.g. from ``layer.get_ray_intersections``) is limited to the
    box (``box_min``, ``box_max``: inclusive voxel corners) and ``volume`` is sampled along it every ``step`` voxels.
    With ``background`` the 1st and ``peak`` the 99th percentile of the box, the point is the first sample (refined
    by linear interpolation) that reaches ``background + fraction * (max along the ray - background)``, i.e. for
    ``fraction=.5`` the half-maximum edge of the first bright structure on the ray, which is the visible surface of
    a blurred object. Returns None if the ray misses the box, or if nothing on it rises above the background by at
    least ``min_contrast`` of ``peak - background``. If the ray starts inside the structure, the point is where it
    enters the box. The result is in the data coordinates of the ray. ``reference`` is ``(background, peak)``
    from :func:`box_reference`, computed here if not given.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    span = clip_ray_to_box(start, end, box_min, box_max)
    if span is None:
        return None
    direction = end - start
    length = np.linalg.norm(direction) * (span[1] - span[0])
    count = max(int(np.ceil(length / step)), 1) + 1
    points = start + np.linspace(span[0], span[1], count)[:, None] * direction
    shape = np.asarray(volume.shape)
    low = np.clip(np.floor(points.min(axis=0)).astype(int) - 1, 0, shape - 1)
    high = np.clip(np.ceil(points.max(axis=0)).astype(int) + 2, 1, shape)
    crop = np.asarray(volume[tuple(slice(a, b) for a, b in zip(low, high))])
    profile = map_coordinates(crop, (points - low).T, order=1, mode="nearest", output=np.float32)

    background, peak = box_reference(volume, box_min, box_max) if reference is None else reference
    top = profile.max()
    if top - background < min_contrast * (peak - background) or top <= background:
        return None
    threshold = background + fraction * (top - background)
    first = int(np.argmax(profile >= threshold))
    if first == 0:
        return points[0]
    below, above = profile[first - 1], profile[first]
    weight = (threshold - below) / (above - below) if above > below else 1.
    return points[first - 1] + weight * (points[first] - points[first - 1])


def pick_on_plane(start, end, plane_position, plane_normal, box_min, box_max) -> Optional[np.ndarray]:
    """Where the line through ``start`` and ``end`` crosses a plane, if that is inside the bounding box.

    Everything is in data coordinates (``box_min``, ``box_max``: inclusive voxel corners). Returns None if the line is
    parallel to the plane or crosses it outside the box.
    """
    start = np.asarray(start, dtype=float)
    direction = np.asarray(end, dtype=float) - start
    normal = np.asarray(plane_normal, dtype=float)
    denominator = direction @ normal
    if abs(denominator) < 1e-12:
        return None
    point = start + ((np.asarray(plane_position, dtype=float) - start) @ normal / denominator) * direction
    if np.any(point < np.asarray(box_min, dtype=float) - .5) or np.any(point > np.asarray(box_max, dtype=float) + .5):
        return None
    return point


def nearest_point_to_ray(points, ray_position, ray_direction, max_distance, max_depth=None) -> Optional[int]:
    """The index of the point closest to a ray, if it is within ``max_distance`` of it, else None.

    All in the same coordinates, e.g. the world coordinates of the cursor ray and of the anchors. Points further along
    the ray than ``max_depth`` (measured from ``ray_position``) are ignored: they are hidden behind what the ray hit.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return None
    direction = np.asarray(ray_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    offsets = points - np.asarray(ray_position, dtype=float)
    depths = offsets @ direction
    distances = np.linalg.norm(offsets - np.outer(depths, direction), axis=1)
    if max_depth is not None:
        distances = np.where(depths <= max_depth, distances, np.inf)
    nearest = int(np.argmin(distances))
    return nearest if distances[nearest] <= max_distance else None
