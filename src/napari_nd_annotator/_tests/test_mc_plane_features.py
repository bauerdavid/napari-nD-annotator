"""The Minimal Contour widget computes the image gradients of the displayed plane when they are needed.

Nothing is precomputed for the volume, nothing is written to disk, and the gradients follow the display order
(``grad_x`` is ``sobel_v`` and ``grad_y`` is ``sobel_h`` of the plane as it is shown).
"""
import glob
import itertools
import os
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pytest
from skimage.filters import gaussian, sobel_h, sobel_v

from napari_nd_annotator._widgets.minimal_contour_widget import (
    ASSYM_GRADIENT, CUSTOM, HIGH_INTENSITY, LOW_INTENSITY, SYM_GRADIENT)
from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
from napari_nd_annotator.minimal_contour import MinimalContourCalculator
from napari_nd_annotator.minimal_contour.plane_features import PlaneFeatureCache, compute_gradients, to_float64_pair

SHAPE = (6, 40, 50)
LIMITS = (0, 1000)


def _process_events():
    from qtpy.QtWidgets import QApplication
    QApplication.processEvents()


@pytest.fixture
def viewer():
    import napari
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()
    _process_events()


@pytest.fixture
def calls(monkeypatch):
    """The arguments of every VispyMinimalContourOverlay.set_calculator_feature call, as copies."""
    recorded = []
    set_calculator_feature = VispyMinimalContourOverlay.set_calculator_feature

    def spy(features, grad_x=None, grad_y=None):
        recorded.append(tuple(None if a is None else np.array(a) for a in (features, grad_x, grad_y)))
        return set_calculator_feature(features, grad_x, grad_y)
    monkeypatch.setattr(VispyMinimalContourOverlay, "set_calculator_feature", staticmethod(spy))
    return recorded


@pytest.fixture
def cache_gets(monkeypatch):
    """The keys of every PlaneFeatureCache.get call."""
    keys = []
    get = PlaneFeatureCache.get

    def spy(self, key, *args, **kwargs):
        keys.append(key)
        return get(self, key, *args, **kwargs)
    monkeypatch.setattr(PlaneFeatureCache, "get", spy)
    return keys


def _scene(viewer, data, feature=ASSYM_GRADIENT, labels_shape=None, enable=True, prepare=None, **image_kwargs):
    from napari_nd_annotator import AnnotatorWidget
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    if prepare is not None:
        prepare(annotator)
    image = viewer.add_image(data, name="image", **image_kwargs)
    shape = tuple(np.shape(data)[:image.ndim]) if labels_shape is None else labels_shape
    labels = viewer.add_labels(np.zeros(shape, dtype=np.uint16), name="labels",
                               metadata={"source_image_layer": image})
    viewer.layers.selection.active = labels
    image.contrast_limits = LIMITS
    mc = annotator.ToolsWidget.minimal_contour_widget
    mc.used_feature_combobox.value = feature
    mc.blur_image_checkbox.value = False
    if enable:
        mc._enable_mc_mode()
    _process_events()
    return SimpleNamespace(annotator=annotator, mc=mc, data=data, image=image, labels=labels)


def _settle(mc):
    """What the debounced dims callback does once scrolling settles."""
    _process_events()
    mc._set_image()
    _process_events()


def _volume(shape, dtype=np.uint16, seed=0):
    return np.random.default_rng(seed).integers(0, 1000, shape).astype(dtype)


def _displayed_plane(data, order, point):
    """The plane napari shows for ``order`` (last two axes are displayed) at ``point``, built from scratch."""
    displayed = sorted(order[-2:])
    index = tuple(slice(None) if axis in displayed else point[axis] for axis in range(data.ndim))
    plane = data[index]
    return plane.T if order[-2] > order[-1] else plane


def _assert_gradients(call, plane):
    _, grad_x, grad_y = call
    plane = plane.astype(float)
    np.testing.assert_array_equal(grad_x, sobel_v(plane))
    np.testing.assert_array_equal(grad_y, sobel_h(plane))


def test_activation_creates_no_temp_files(viewer):
    pattern = os.path.join(tempfile.gettempdir(), "*_%d_nd_annotator" % os.getpid())
    scene = _scene(viewer, _volume(SHAPE), enable=False)
    assert glob.glob(pattern) == []
    scene.mc._enable_mc_mode()
    viewer.dims.set_point(0, 3)
    _settle(scene.mc)
    viewer.dims.order = (1, 0, 2)
    _settle(scene.mc)
    assert glob.glob(pattern) == []
    assert not hasattr(scene.mc, "feature_manager")


def test_cache_stays_within_its_limit_while_scrolling_100_slices(viewer, calls):
    data = _volume((100, 64, 64))
    scene = _scene(viewer, data, enable=False)
    mc = scene.mc
    plane_bytes = 2 * 64 * 64 * 4
    mc._plane_features = PlaneFeatureCache(max_bytes=3 * plane_bytes + 10)
    mc._enable_mc_mode()
    n = len(calls)
    for z in range(100):
        viewer.dims.set_point(0, z)
        _settle(mc)
        assert mc._plane_features.nbytes <= mc._plane_features.max_bytes
        assert len(mc._plane_features) <= 3
    assert len(calls) - n >= 100
    _assert_gradients(calls[-1], data[99])
    assert len(mc._plane_features) == 3


def test_scrolling_back_hits_the_cache(viewer, calls, monkeypatch):
    import napari_nd_annotator.minimal_contour.plane_features as plane_features
    computed = []
    compute = plane_features.compute_gradients

    def counting(plane, rgb=False):
        computed.append(1)
        return compute(plane, rgb)
    data = _volume(SHAPE)
    scene = _scene(viewer, data)
    monkeypatch.setattr(plane_features, "compute_gradients", counting)
    for z in (1, 2, 1, 2, 1):
        viewer.dims.set_point(0, z)
        _settle(scene.mc)
        _assert_gradients(calls[-1], data[z])
    assert len(computed) <= 2


@pytest.mark.parametrize("feature", [ASSYM_GRADIENT, SYM_GRADIENT, HIGH_INTENSITY, LOW_INTENSITY])
def test_feature_modes_reach_the_calculator(viewer, calls, feature):
    data = _volume(SHAPE)
    scene = _scene(viewer, data, feature, enable=False)
    viewer.dims.set_point(0, 3)
    scene.mc._enable_mc_mode()
    _settle(scene.mc)
    plane = data[3].astype(float)
    image = np.clip(plane / LIMITS[1], 0, 1)
    features, grad_x, grad_y = calls[-1]
    if feature == SYM_GRADIENT:
        assert grad_x is None and grad_y is None
        magnitude = np.hypot(sobel_v(plane), sobel_h(plane)) / LIMITS[1]
        expected = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        assert features.shape == plane.shape
        np.testing.assert_allclose(features, expected, atol=1e-9)
        return
    expected_image = 1 - image if feature == LOW_INTENSITY else image
    assert features.shape == plane.shape + (3,)
    for channel in range(3):
        np.testing.assert_allclose(features[..., channel], expected_image, atol=1e-12)
    _assert_gradients(calls[-1], data[3])


def test_custom_feature_script_still_runs(viewer, calls, cache_gets, monkeypatch):
    data = _volume(SHAPE)
    scene = _scene(viewer, data, enable=False)
    mc = scene.mc
    editor = mc.feature_editor
    # running a script stores it in the user's settings: keep that out of the test
    store = {}
    monkeypatch.setattr(editor, "text_settings",
                        SimpleNamespace(value=lambda key, default="": store.get(key, default),
                                        setValue=lambda key, value: store.__setitem__(key, value)))
    editor.code_editor.document().setPlainText("features = image[..., 0] * 0.5")
    mc.used_feature_combobox.value = CUSTOM
    viewer.dims.set_point(0, 2)
    mc._enable_mc_mode()
    deadline = time.time() + 20
    while not any(c[0] is not None for c in calls) and time.time() < deadline:
        _process_events()
        time.sleep(0.01)
    assert calls[-1][0] is not None, "the custom feature script did not deliver features"
    features, grad_x, grad_y = calls[-1]
    expected = np.clip(data[2].astype(float) / LIMITS[1], 0, 1) * 0.5
    np.testing.assert_allclose(features[..., 0], expected, atol=1e-12)
    assert grad_x is None and grad_y is None
    assert cache_gets == []


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_gradients_follow_the_display_order_3d(viewer, calls, order):
    data = _volume((5, 7, 9))
    scene = _scene(viewer, data)
    mc = scene.mc
    point = (2, 3, 4)
    for axis, value in enumerate(point):
        viewer.dims.set_point(axis, value)
    viewer.dims.order = order
    _settle(mc)
    plane = _displayed_plane(data, order, point)
    features, _, _ = calls[-1]
    assert features.shape[:2] == plane.shape
    _assert_gradients(calls[-1], plane)


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1), (0, 3, 2, 1), (3, 0, 1, 2)])
def test_gradients_follow_the_display_order_4d(viewer, calls, monkeypatch, order):
    data = _volume((3, 5, 7, 9))

    def ignore_minimal_surface(annotator):
        # the Minimal Surface widget cannot handle 4D layers (it raises while the labels layer is added)
        minimal_surface = annotator.ToolsWidget.minimal_surface_widget
        if minimal_surface is None:  # not available with this napari version
            return
        clipping_widget = minimal_surface.ImageSliceWidget.clipping_widget
        monkeypatch.setattr(type(clipping_widget), "_update_ranges", lambda self: None)
    scene = _scene(viewer, data, prepare=ignore_minimal_surface)
    mc = scene.mc
    point = (1, 2, 3, 4)
    for axis, value in enumerate(point):
        viewer.dims.set_point(axis, value)
    viewer.dims.order = order
    _settle(mc)
    plane = _displayed_plane(data, order, point)
    assert calls[-1][0].shape[:2] == plane.shape
    _assert_gradients(calls[-1], plane)


def test_rgb_image(viewer, calls):
    data = _volume(SHAPE + (3,), np.uint8)
    data[..., 1] = 0  # an empty channel is skipped
    scene = _scene(viewer, data, rgb=True)
    viewer.dims.set_point(0, 2)
    _settle(scene.mc)
    features, grad_x, grad_y = calls[-1]
    assert features.shape == SHAPE[1:] + (3,)
    expected_x, expected_y = compute_gradients(data[2], rgb=True)
    np.testing.assert_allclose(grad_x, expected_x)
    np.testing.assert_allclose(grad_y, expected_y)
    assert np.abs(grad_x).max() > 0
    assert scene.image.data is data


@pytest.mark.parametrize("kind", ["dask", "memmap"])
def test_lazy_data_sources(viewer, calls, tmp_path, kind):
    values = _volume(SHAPE)
    if kind == "dask":
        da = pytest.importorskip("dask.array")
        data = da.from_array(values.copy(), chunks=(1,) + SHAPE[1:])
    else:
        path = str(tmp_path / "image.dat")
        mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=SHAPE)
        mm[:] = values
        mm.flush()
        del mm
        data = np.memmap(path, dtype=np.uint16, mode="r", shape=SHAPE)
    scene = _scene(viewer, data)
    for z in (1, 4):
        viewer.dims.set_point(0, z)
        _settle(scene.mc)
        _assert_gradients(calls[-1], values[z])
    viewer.dims.order = (2, 0, 1)
    _settle(scene.mc)
    point = tuple(viewer.dims.current_step)
    _assert_gradients(calls[-1], _displayed_plane(values, (2, 0, 1), point))
    assert scene.image.data is data


@pytest.mark.parametrize("sigma", [1.5, 3.])
def test_gradients_are_blurred_once_when_blur_is_on(viewer, calls, sigma):
    data = _volume(SHAPE)
    scene = _scene(viewer, data)
    mc = scene.mc
    viewer.dims.set_point(0, 3)
    _settle(mc)
    plane = data[3].astype(float)
    np.testing.assert_array_equal(calls[-1][1], sobel_v(plane))

    mc.blur_sigma_slider.value = sigma
    mc.blur_image_checkbox.value = True
    _settle(mc)

    def blur(a):
        return gaussian(a, sigma=sigma, preserve_range=True, truncate=2.)
    _, grad_x, grad_y = calls[-1]
    np.testing.assert_allclose(grad_x, blur(sobel_v(plane)), atol=1e-6)
    np.testing.assert_allclose(grad_y, blur(sobel_h(plane)), atol=1e-6)
    assert np.abs(grad_x - blur(blur(sobel_v(plane)))).max() > 1e-3

    mc.blur_image_checkbox.value = False
    _settle(mc)
    np.testing.assert_array_equal(calls[-1][1], sobel_v(plane))


def test_replacing_the_image_data_gives_fresh_gradients(viewer, calls):
    data = _volume(SHAPE)
    scene = _scene(viewer, data)
    viewer.dims.set_point(0, 2)
    _settle(scene.mc)
    _assert_gradients(calls[-1], data[2])
    new = _volume(SHAPE, seed=5)
    scene.image.data = new
    _settle(scene.mc)
    _assert_gradients(calls[-1], new[2])


def test_removing_the_image_layer_clears_its_planes(viewer, calls):
    scene = _scene(viewer, _volume(SHAPE))
    mc = scene.mc
    for z in range(3):
        viewer.dims.set_point(0, z)
        _settle(mc)
    assert len(mc._plane_features) >= 3
    viewer.layers.remove(scene.image)
    _process_events()
    assert len(mc._plane_features) == 0
    # the labels layer still refers to the removed image: nothing may be cached for it again
    mc._set_image()
    _process_events()
    assert len(mc._plane_features) == 0
    assert mc._get_gradients(scene.image) == (None, None)


def test_no_gradients_in_3d_display_or_out_of_range(viewer, monkeypatch):
    scene = _scene(viewer, _volume(SHAPE))
    mc, image = scene.mc, scene.image
    viewer.dims.set_point(0, 2)
    _settle(mc)
    grad_x, grad_y = mc._get_gradients(image)
    assert grad_x.shape == grad_y.shape == SHAPE[1:] and grad_x.dtype == np.float32

    import napari_nd_annotator._widgets.minimal_contour_widget as widget
    for index in (SHAPE[0], -1):
        monkeypatch.setattr(widget, "layer_slice_indices", lambda layer, index=index: (index, slice(None), slice(None)))
        assert mc._get_gradients(image) == (None, None)
    monkeypatch.undo()

    viewer.dims.ndisplay = 3
    _process_events()
    assert mc._get_gradients(image) == (None, None)


def test_the_calculator_contour_does_not_depend_on_the_gradient_precision():
    # a noisy float image with a bright disc: float32 gradients (cached) against float64 ones (before)
    yy, xx = np.mgrid[:90, :100]
    rng = np.random.default_rng(0)
    plane = 200 * ((yy - 45) ** 2 + (xx - 50) ** 2 < 25 ** 2) + rng.random((90, 100)) * 20
    image = np.repeat(np.clip(plane / 255, 0, 1)[..., None], 3, axis=2)
    gradients32 = to_float64_pair(compute_gradients(plane))
    gradients64 = (sobel_v(plane), sobel_h(plane))
    assert gradients32[0].dtype == gradients64[0].dtype == np.float64
    assert not np.array_equal(gradients32[0], gradients64[0])  # not trivially the same input
    anchors = np.array([[20., 50.], [50., 76.], [72., 40.]])  # as (y, x)
    results = []
    for gradient_x, gradient_y in (gradients32, gradients64):
        calculator = MinimalContourCalculator(None, 3)
        calculator.set_method(0)
        calculator.set_param(5)
        calculator.set_image(image, gradient_x, gradient_y)
        calculator.set_boundaries(0, 0, 100, 90)
        results.append(calculator.run(anchors, True, True, True))
    segments32, segments64 = results
    assert len(segments32) == len(segments64) == 3
    assert sum(len(s) for s in segments32) > 50
    for a, b in zip(segments32, segments64):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
