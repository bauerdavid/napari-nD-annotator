"""Image gradients of one displayed plane, computed on demand and kept in a small byte-bounded LRU cache.

The minimal contour calculator needs the x and y derivative of the plane that is on screen. A pair costs a few
milliseconds per plane, so nothing is precomputed for the whole volume: a plane is processed when it is asked
for, and the most recent ones stay in the cache. This module has no Qt or napari dependency.
"""
import os
import threading
from collections import OrderedDict
from typing import Callable, Hashable, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

CACHE_ENV_VAR = "NAPARI_ND_ANNOTATOR_FEATURE_CACHE_MB"
DEFAULT_CACHE_BYTES = 64 * 1024 * 1024


def default_cache_bytes() -> int:
    """DEFAULT_CACHE_BYTES, or the size in MB given by the environment variable (when it is a valid number)."""
    try:
        megabytes = float(os.environ[CACHE_ENV_VAR])
    except (KeyError, ValueError):
        return DEFAULT_CACHE_BYTES
    if not megabytes >= 0:  # also rejects nan
        return DEFAULT_CACHE_BYTES
    return int(megabytes * 1024 * 1024)


def compute_gradients(plane: np.ndarray, rgb: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """The x and y derivative (along axis 1 and axis 0) of a 2D plane, or of an HxWxC plane if ``rgb``, as float32.

    For a gray plane these equal ``skimage.filters.sobel_v`` and ``sobel_h``.
    """
    if not rgb:
        # transposed views are not C-contiguous, which cv2 would silently copy anyway
        plane = np.ascontiguousarray(plane, dtype=np.float32)
        grad_x = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3, scale=0.25, borderType=cv2.BORDER_REFLECT)
        grad_y = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3, scale=0.25, borderType=cv2.BORDER_REFLECT)
        return grad_x, grad_y
    channels_v = []
    channels_h = []
    for channel in range(3):
        c = plane[..., channel].astype(float)
        if np.any(c):
            channels_v.append(gaussian_filter(c, 2, (0, 1)))
            channels_h.append(gaussian_filter(c, 2, (1, 0)))
    if not channels_v:
        shape = plane.shape[:2]
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    return sum(channels_v).astype(np.float32), sum(channels_h).astype(np.float32)


def to_float64_pair(pair):
    """float64 copies of a gradient pair, as the (float64 only) minimal contour calculator needs them."""
    return tuple(None if a is None else np.array(a, dtype=np.float64) for a in pair)


class PlaneFeatureCache:
    """Thread-safe LRU cache of gradient pairs, bounded by the bytes they take.

    Keys are hashable tuples whose first item is the id of the owning layer (see ``invalidate``).
    Cached arrays are read-only: callers must copy before modifying.
    """

    def __init__(self, max_bytes: Optional[int] = None):
        self.max_bytes = default_cache_bytes() if max_bytes is None else int(max_bytes)
        self._entries: "OrderedDict[Hashable, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._nbytes = 0
        self._lock = threading.Lock()

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        with self._lock:
            return key in self._entries

    def get(self, key: Hashable, plane_provider: Callable[[], np.ndarray], rgb: bool = False):
        """The gradient pair for ``key``; ``plane_provider()`` is only called (outside the lock) on a miss."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry
        entry = compute_gradients(plane_provider(), rgb)
        for a in entry:
            a.setflags(write=False)
        size = sum(a.nbytes for a in entry)
        if size > self.max_bytes:
            return entry  # larger than the whole cache: returned, but would only evict everything else
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:  # another thread computed the same plane meanwhile
                self._nbytes -= sum(a.nbytes for a in old)
            self._entries[key] = entry
            self._nbytes += size
            while self._nbytes > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._nbytes -= sum(a.nbytes for a in evicted)
        return entry

    def invalidate(self, layer_id: int):
        """Drop every entry whose key starts with ``layer_id``."""
        with self._lock:
            for key in [k for k in self._entries if k[0] == layer_id]:
                self._nbytes -= sum(a.nbytes for a in self._entries.pop(key))

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._nbytes = 0
