"""Lazy per-plane gradients and their LRU cache (no viewer needed)."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel_h, sobel_v

from napari_nd_annotator.minimal_contour import plane_features
from napari_nd_annotator.minimal_contour.plane_features import (
    CACHE_ENV_VAR, DEFAULT_CACHE_BYTES, PlaneFeatureCache, compute_gradients, default_cache_bytes, to_float64_pair)


def _plane(dtype, shape=(37, 53), seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, np.iinfo(dtype).max // 2, shape).astype(dtype)
    return rng.random(shape).astype(dtype) * 100


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(40, 40), (37, 53), (64, 5)])
def test_gray_gradients_equal_skimage_sobel(dtype, shape):
    plane = _plane(dtype, shape)
    grad_x, grad_y = compute_gradients(plane)
    assert grad_x.dtype == grad_y.dtype == np.float32
    assert grad_x.shape == grad_y.shape == shape
    expected_x, expected_y = sobel_v(plane.astype(float)), sobel_h(plane.astype(float))
    if np.issubdtype(dtype, np.integer):
        np.testing.assert_array_equal(grad_x, expected_x)
        np.testing.assert_array_equal(grad_y, expected_y)
    else:
        np.testing.assert_allclose(grad_x, expected_x, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(grad_y, expected_y, rtol=1e-5, atol=1e-4)


def test_gray_gradients_of_a_transposed_view():
    plane = _plane(np.uint16, (30, 45))
    grad_x, grad_y = compute_gradients(plane.T)
    np.testing.assert_array_equal(grad_x, sobel_v(plane.T.astype(float)))
    np.testing.assert_array_equal(grad_y, sobel_h(plane.T.astype(float)))


def _reference_rgb(plane):
    channels = [plane[..., i].astype(float) for i in range(3)]
    channels = [c for c in channels if np.any(c)]
    return (sum(gaussian_filter(c, 2, (0, 1)) for c in channels),
            sum(gaussian_filter(c, 2, (1, 0)) for c in channels))


@pytest.mark.parametrize("empty_channel", [None, 0, 2])
def test_rgb_gradients_equal_the_reference_math(empty_channel):
    plane = np.random.default_rng(1).integers(0, 255, (33, 41, 3)).astype(np.uint8)
    if empty_channel is not None:
        plane[..., empty_channel] = 0
    grad_x, grad_y = compute_gradients(plane, rgb=True)
    expected_x, expected_y = _reference_rgb(plane)
    assert grad_x.dtype == grad_y.dtype == np.float32
    np.testing.assert_allclose(grad_x, expected_x, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(grad_y, expected_y, rtol=1e-5, atol=1e-4)


def test_rgb_alpha_channel_is_ignored():
    rgb = np.random.default_rng(2).integers(0, 255, (20, 25, 3)).astype(np.uint8)
    rgba = np.concatenate([rgb, np.full((20, 25, 1), 255, np.uint8)], axis=-1)
    for a, b in zip(compute_gradients(rgb, rgb=True), compute_gradients(rgba, rgb=True)):
        np.testing.assert_array_equal(a, b)


def test_rgb_all_zero_plane_gives_zero_gradients():
    grad_x, grad_y = compute_gradients(np.zeros((10, 12, 3), np.uint8), rgb=True)
    assert grad_x.shape == grad_y.shape == (10, 12)
    assert not grad_x.any() and not grad_y.any()


def test_to_float64_pair_copies():
    pair = compute_gradients(_plane(np.uint8))
    grad_x, grad_y = to_float64_pair(pair)
    assert grad_x.dtype == grad_y.dtype == np.float64
    assert grad_x.flags.writeable
    np.testing.assert_array_equal(grad_x, pair[0])
    assert to_float64_pair((None, None)) == (None, None)


class _Provider:
    def __init__(self, plane):
        self.plane = plane
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.plane


def _entry_bytes(shape=(16, 16)):
    return 2 * int(np.prod(shape)) * 4


def test_hit_returns_the_cached_entry_without_calling_the_provider():
    cache = PlaneFeatureCache(10 * _entry_bytes())
    provider = _Provider(_plane(np.uint8, (16, 16)))
    first = cache.get(("layer", 0), provider)
    second = cache.get(("layer", 0), provider)
    assert provider.calls == 1
    assert first[0] is second[0] and first[1] is second[1]
    assert len(cache) == 1 and cache.nbytes == _entry_bytes()
    assert ("layer", 0) in cache


def test_cached_arrays_are_read_only():
    cache = PlaneFeatureCache(10 * _entry_bytes())
    grad_x, grad_y = cache.get(("layer", 0), _Provider(_plane(np.uint8, (16, 16))))
    with pytest.raises(ValueError):
        grad_x[0, 0] = 1
    with pytest.raises(ValueError):
        grad_y[0, 0] = 1


def test_least_recently_used_entry_is_evicted_and_the_byte_bound_holds():
    cache = PlaneFeatureCache(3 * _entry_bytes())
    providers = [_Provider(_plane(np.uint8, (16, 16), seed=i)) for i in range(5)]
    for i in range(3):
        cache.get(("l", i), providers[i])
    cache.get(("l", 0), providers[0])  # 0 becomes the most recent, 1 the oldest
    cache.get(("l", 3), providers[3])
    assert ("l", 1) not in cache
    assert all(("l", i) in cache for i in (0, 2, 3))
    for i in range(5):
        cache.get(("l", i), providers[i])
        assert cache.nbytes <= cache.max_bytes
        assert len(cache) <= 3
    assert cache.nbytes == 3 * _entry_bytes()


def test_oversize_entry_is_returned_but_not_cached():
    cache = PlaneFeatureCache(_entry_bytes() - 1)
    provider = _Provider(_plane(np.uint8, (16, 16)))
    grad_x, grad_y = cache.get(("l", 0), provider)
    assert grad_x.shape == (16, 16)
    assert len(cache) == 0 and cache.nbytes == 0
    cache.get(("l", 0), provider)
    assert provider.calls == 2


def test_oversize_entry_does_not_evict_the_others():
    cache = PlaneFeatureCache(2 * _entry_bytes((16, 16)))
    cache.get(("l", 0), _Provider(_plane(np.uint8, (16, 16))))
    cache.get(("l", 1), _Provider(_plane(np.uint8, (64, 64))))
    assert ("l", 0) in cache and ("l", 1) not in cache


def test_zero_size_cache_never_stores():
    cache = PlaneFeatureCache(0)
    cache.get(("l", 0), _Provider(_plane(np.uint8, (4, 4))))
    assert len(cache) == 0


def test_invalidate_drops_only_the_given_layer():
    cache = PlaneFeatureCache(10 * _entry_bytes())
    for layer in (1, 2):
        for z in range(3):
            cache.get((layer, z), _Provider(_plane(np.uint8, (16, 16), seed=z)))
    cache.invalidate(1)
    assert len(cache) == 3 and cache.nbytes == 3 * _entry_bytes()
    assert all((2, z) in cache for z in range(3))
    cache.invalidate(99)
    assert len(cache) == 3


def test_clear():
    cache = PlaneFeatureCache(10 * _entry_bytes())
    cache.get((1, 0), _Provider(_plane(np.uint8, (16, 16))))
    cache.clear()
    assert len(cache) == 0 and cache.nbytes == 0


def test_rgb_flag_selects_the_rgb_path():
    cache = PlaneFeatureCache(10 * _entry_bytes())
    plane = np.random.default_rng(3).integers(1, 255, (16, 16, 3)).astype(np.uint8)
    grad_x, _ = cache.get((1, 0), _Provider(plane), rgb=True)
    np.testing.assert_allclose(grad_x, _reference_rgb(plane)[0], rtol=1e-5, atol=1e-4)


def test_default_size_and_environment_variable(monkeypatch):
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    assert default_cache_bytes() == DEFAULT_CACHE_BYTES == 64 * 1024 * 1024
    assert PlaneFeatureCache().max_bytes == DEFAULT_CACHE_BYTES
    monkeypatch.setenv(CACHE_ENV_VAR, "8")
    assert PlaneFeatureCache().max_bytes == 8 * 1024 * 1024
    monkeypatch.setenv(CACHE_ENV_VAR, "0.5")
    assert PlaneFeatureCache().max_bytes == 512 * 1024
    for bad in ("", "abc", "-3", "nan"):
        monkeypatch.setenv(CACHE_ENV_VAR, bad)
        assert default_cache_bytes() == DEFAULT_CACHE_BYTES
    assert PlaneFeatureCache(1234).max_bytes == 1234


def test_concurrent_gets_stay_consistent():
    cache = PlaneFeatureCache(4 * _entry_bytes())
    planes = [_plane(np.uint8, (16, 16), seed=i) for i in range(10)]
    expected = [compute_gradients(p) for p in planes]

    def worker(seed):
        rng = np.random.default_rng(seed)
        for _ in range(200):
            i = int(rng.integers(0, len(planes)))
            grad_x, grad_y = cache.get((1, i), lambda: planes[i])
            np.testing.assert_array_equal(grad_x, expected[i][0])
            np.testing.assert_array_equal(grad_y, expected[i][1])
            assert cache.nbytes <= cache.max_bytes
            if seed == 0 and rng.random() < 0.05:
                cache.invalidate(1)

    with ThreadPoolExecutor(8) as pool:
        for future in [pool.submit(worker, seed) for seed in range(8)]:
            future.result()
    assert cache.nbytes == sum(sum(a.nbytes for a in e) for e in cache._entries.values())
    assert cache.nbytes <= cache.max_bytes


def test_module_does_not_import_qt_or_napari():
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, %r);"
            "import importlib.util as u;"
            "s = u.spec_from_file_location('pf', %r); m = u.module_from_spec(s); s.loader.exec_module(m);"
            "bad = [n for n in sys.modules if n.split('.')[0] in ('napari', 'qtpy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6')];"
            "print(bad)") % ("", plane_features.__file__)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
