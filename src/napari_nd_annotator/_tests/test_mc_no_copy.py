"""The Minimal Contour widget never copies or modifies the source image layer's data.

Blurring is applied once, to the plane handed to the minimal contour calculator; the image layer keeps the
user's array (numpy, dask, np.memmap), so Minimal Surface and the projections see the unmodified image.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from skimage.filters import gaussian

from napari_nd_annotator._widgets.minimal_contour_widget import HIGH_INTENSITY, DEMO_SIZE
from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay

SHAPE = (6, 40, 50)


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
def calculator_calls(monkeypatch):
    """The arguments of every VispyMinimalContourOverlay.set_calculator_feature call."""
    calls = []
    set_calculator_feature = VispyMinimalContourOverlay.set_calculator_feature

    def spy(features, grad_x=None, grad_y=None):
        # copies: the arrays given to the calculator must not change when the test inspects them later
        calls.append(tuple(None if a is None else np.array(a) for a in (features, grad_x, grad_y)))
        return set_calculator_feature(features, grad_x, grad_y)
    monkeypatch.setattr(VispyMinimalContourOverlay, "set_calculator_feature", staticmethod(spy))
    return calls


def _scene(viewer, data, labels_shape, **image_kwargs):
    from napari_nd_annotator import AnnotatorWidget
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    image = viewer.add_image(data, name="image", **image_kwargs)
    labels = viewer.add_labels(np.zeros(labels_shape, dtype=np.uint16), name="labels",
                               metadata={"source_image_layer": image})
    viewer.layers.selection.active = labels
    _process_events()
    return SimpleNamespace(annotator=annotator, mc=annotator.ToolsWidget.minimal_contour_widget,
                           data=data, image=image, labels=labels)


def _volume(kind, tmp_path):
    values = np.random.default_rng(0).integers(0, 1000, SHAPE).astype(np.uint16)
    if kind == "numpy":
        data = values.copy()
    elif kind == "dask":
        da = pytest.importorskip("dask.array")
        data = da.from_array(values.copy(), chunks=(1,) + SHAPE[1:])
    else:
        path = str(tmp_path / "image.dat")
        mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=SHAPE)
        mm[:] = values
        mm.flush()
        del mm
        data = np.memmap(path, dtype=np.uint16, mode="r", shape=SHAPE)
    return data, values


def _settle(mc):
    """What the debounced dims callback does once scrolling settles."""
    _process_events()
    mc._set_image()
    _process_events()


@pytest.mark.parametrize("kind", ["numpy", "dask", "memmap"])
def test_image_data_is_never_copied_or_modified(viewer, tmp_path, kind):
    data, values = _volume(kind, tmp_path)
    scene = _scene(viewer, data, SHAPE)
    mc, image = scene.mc, scene.image
    ms = scene.annotator.ToolsWidget.minimal_surface_widget

    def check():
        assert image.data is data
        np.testing.assert_array_equal(np.asarray(image.data), values)

    assert not hasattr(mc, "_orig_image")
    viewer.dims.set_point(0, 2)
    mc._enable_mc_mode()
    _settle(mc)
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]
    check()

    mc.blur_image_checkbox.value = True
    _settle(mc)
    check()
    for sigma in (0.5, 3.):
        mc.blur_sigma_slider.value = sigma
        _settle(mc)
        check()

    for z in (3, 5, 0):
        viewer.dims.set_point(0, z)
        _settle(mc)
        check()

    viewer.dims.order = (1, 0, 2)
    _settle(mc)
    assert VispyMinimalContourOverlay._features_shape == (SHAPE[0], SHAPE[2])
    check()
    viewer.dims.order = (0, 1, 2)
    _settle(mc)
    check()

    mc.blur_image_checkbox.value = False
    _settle(mc)
    check()

    # Minimal Surface reads image.data of the same layer
    if ms is not None:
        ms.image_layer = image
        _process_events()
        assert ms.image_layer.data is data

    viewer.layers.selection.active = image
    _process_events()
    check()
    viewer.layers.selection.active = scene.labels
    mc.blur_image_checkbox.value = True
    _settle(mc)
    check()

    viewer.layers.remove(scene.labels)
    _process_events()
    assert mc.labels_layer is None
    check()


def test_the_calculator_image_is_blurred_once(viewer, calculator_calls):
    yy, xx = np.mgrid[:128, :128]
    rng = np.random.default_rng(0)
    base = 3000 * np.exp(-((yy - 60) ** 2 + (xx - 70) ** 2) / (2 * 15 ** 2)) + rng.normal(0, 200, (128, 128))
    data = np.stack([base + 10 * i for i in range(6)]).clip(0, 65535).astype(np.uint16)
    scene = _scene(viewer, data, data.shape)
    mc = scene.mc
    scene.image.contrast_limits = (0, 8000)
    mc.used_feature_combobox.value = HIGH_INTENSITY
    viewer.dims.set_point(0, 3)
    mc._enable_mc_mode()
    _settle(mc)

    plane = data[3].astype(float)

    def norm(a):
        return np.clip(a / 8000., 0, 1)

    def blur(a):
        return gaussian(a, sigma=3., preserve_range=True, truncate=2.)

    def calculator_image():
        features, grad_x, grad_y = calculator_calls[-1]
        assert features.shape == plane.shape + (3,)
        assert grad_x is not None and grad_y is not None
        return features[..., 0]

    np.testing.assert_allclose(calculator_image(), norm(plane), atol=1e-9)

    mc.blur_sigma_slider.value = 3.
    mc.blur_image_checkbox.value = True
    _settle(mc)
    once = norm(blur(plane))
    np.testing.assert_allclose(calculator_image(), once, atol=1e-6)
    # before, the plane was also blurred inside the image layer: the effective sigma was sqrt(2) * sigma
    assert np.abs(calculator_image() - blur(once)).max() > 1e-3
    np.testing.assert_array_equal(scene.image.data[3], data[3])


def test_multiscale_image_does_not_raise(viewer):
    values = np.random.default_rng(0).integers(0, 1000, SHAPE).astype(np.uint16)
    pyramid = [values, values[:, ::2, ::2]]
    scene = _scene(viewer, pyramid, SHAPE, multiscale=True)
    mc = scene.mc
    assert scene.image.multiscale
    with pytest.warns(UserWarning, match="multiscale"):
        mc._enable_mc_mode()
        _process_events()
    # without features the overlay ignores the mouse, so the tool does nothing on this layer
    assert VispyMinimalContourOverlay._features_shape is None

    mc.blur_image_checkbox.value = True
    mc.blur_sigma_slider.value = 2.
    viewer.dims.set_point(0, 3)
    _settle(mc)
    assert VispyMinimalContourOverlay._features_shape is None
    assert scene.image.data[0] is values
    assert np.asarray(mc.demo_image_widget.value).shape[-1] == 3


def test_rgb_image(viewer, calculator_calls):
    data = np.random.default_rng(0).integers(0, 255, SHAPE + (3,)).astype(np.uint8)
    values = data.copy()
    scene = _scene(viewer, data, SHAPE)
    mc = scene.mc
    assert scene.image.rgb
    viewer.dims.set_point(0, 2)
    mc._enable_mc_mode()
    mc.blur_image_checkbox.value = True
    mc.blur_sigma_slider.value = 2.
    _settle(mc)
    features, grad_x, grad_y = calculator_calls[-1]
    assert features.shape == SHAPE[1:] + (3,)
    assert grad_x.shape == SHAPE[1:]
    assert scene.image.data is data
    np.testing.assert_array_equal(data, values)
    assert np.asarray(mc.demo_image_widget.value).shape[-1] == 3


@pytest.mark.parametrize("kind", ["numpy", "dask", "rgb", "large"])
def test_blur_preview(viewer, tmp_path, kind):
    if kind == "rgb":
        data = np.random.default_rng(0).integers(0, 255, SHAPE + (3,)).astype(np.uint8)
        labels_shape = SHAPE
    elif kind == "large":
        # larger than the preview: the centre is cropped
        data = np.random.default_rng(0).integers(0, 1000, (3, 250, 260)).astype(np.uint16)
        labels_shape = data.shape
    else:
        data, _ = _volume(kind, tmp_path)
        labels_shape = SHAPE
    scene = _scene(viewer, data, labels_shape)
    mc = scene.mc
    viewer.dims.set_point(0, 1)
    _process_events()
    mc.blur_image_checkbox.value = True
    previews = []
    for sigma in (0.5, 2.5):
        mc.blur_sigma_slider.value = sigma
        previews.append(np.asarray(mc.demo_image_widget.value).copy())
    assert previews[0].shape[-1] == 3
    assert max(previews[0].shape[:2]) <= max(DEMO_SIZE, *labels_shape[1:])
    assert not np.array_equal(*previews)
    assert scene.image.data is data
