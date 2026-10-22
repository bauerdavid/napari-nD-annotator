"""The Minimal Contour widget only processes the source image while the minimal contour tool is on.

Painting and scrolling with the other labels tools must not copy the image, compute image features or
write feature files: on large volumes that work froze napari on every slice change.
"""
import glob
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers.labels._labels_constants import Mode

from napari_nd_annotator._helper_functions import viewer_qt_viewer
from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
from napari_nd_annotator.minimal_contour.plane_features import PlaneFeatureCache

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
def feature_calls(monkeypatch):
    """The ids of the layers whose plane features were asked for."""
    calls = []
    get = PlaneFeatureCache.get

    def spy(self, key, *args, **kwargs):
        calls.append(key[0])
        return get(self, key, *args, **kwargs)
    monkeypatch.setattr(PlaneFeatureCache, "get", spy)
    return calls


@pytest.fixture
def scene(viewer, feature_calls):
    from napari_nd_annotator import AnnotatorWidget
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    data = np.random.default_rng(0).integers(0, 1000, SHAPE).astype(np.uint16)
    image = viewer.add_image(data, name="image")
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels", metadata={"source_image_layer": image})
    viewer.layers.selection.active = labels
    _process_events()
    return SimpleNamespace(mc=annotator.ToolsWidget.minimal_contour_widget, data=data, values=data.copy(),
                           image=image, labels=labels)


def _assert_image_untouched(scene):
    assert scene.image.data is scene.data
    np.testing.assert_array_equal(scene.data, scene.values)


def _temp_folders():
    """The feature folders of this process (the memmap based features of older versions used to make them)."""
    pattern = os.path.join(tempfile.gettempdir(), "*_%d_nd_annotator" % os.getpid())
    return glob.glob(pattern)


def _assert_idle(scene, feature_calls):
    assert feature_calls == []
    assert scene.image.data is scene.data
    assert _temp_folders() == []
    assert VispyMinimalContourOverlay._features_shape is None


def _scroll(viewer, mc, steps):
    for z in steps:
        viewer.dims.set_point(0, z)
        mc._set_image()  # what the debounced dims callback does once scrolling settles
        _process_events()


def _enable_with_button(viewer, labels):
    viewer_qt_viewer(viewer).controls.widgets[labels].mc_button.click()
    _process_events()


def test_painting_and_scrolling_without_the_tool_does_no_image_work(viewer, scene, feature_calls):
    mc = scene.mc
    assert mc.labels_layer is scene.labels and mc.image_layer is scene.image
    _assert_idle(scene, feature_calls)

    scene.labels.mode = Mode.PAINT
    _scroll(viewer, mc, range(SHAPE[0]))
    _assert_idle(scene, feature_calls)

    viewer.dims.order = (1, 0, 2)
    _process_events()
    viewer.dims.order = (0, 1, 2)
    viewer.dims.ndisplay = 3
    viewer.dims.ndisplay = 2
    _process_events()
    _assert_idle(scene, feature_calls)

    mc.blur_image_checkbox.value = True
    mc.used_feature_combobox.value = "Gradient [symmetric]"
    mc.param_spinbox.value = 7
    _process_events()
    _assert_idle(scene, feature_calls)


@pytest.mark.parametrize("enable", ["button", "action"])
def test_turning_the_tool_on_prepares_the_features(viewer, scene, feature_calls, enable):
    viewer.dims.set_point(0, 3)
    _process_events()
    if enable == "button":
        _enable_with_button(viewer, scene.labels)
    else:
        scene.mc._enable_mc_mode()
        _process_events()
    assert scene.labels._overlays["minimal_contour"].enabled
    assert id(scene.image) in feature_calls
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]
    assert _temp_folders() == []


def test_turning_the_tool_off_leaves_nothing_behind(viewer, scene, feature_calls):
    mc = scene.mc
    mc.blur_image_checkbox.value = True
    viewer.dims.set_point(0, 2)
    _enable_with_button(viewer, scene.labels)
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]
    _assert_image_untouched(scene)

    scene.labels.mode = Mode.PAINT
    _process_events()
    assert not scene.labels._overlays["minimal_contour"].enabled
    assert VispyMinimalContourOverlay._features_shape is None
    _assert_image_untouched(scene)

    n_calls = len(feature_calls)
    _scroll(viewer, mc, [4, 5])
    assert len(feature_calls) == n_calls
    _assert_image_untouched(scene)

    # on again, on another slice and with a transposed display order
    mc._enable_mc_mode()
    _process_events()
    assert len(feature_calls) > n_calls
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]
    viewer.dims.order = (2, 0, 1)
    _process_events()
    assert VispyMinimalContourOverlay._features_shape == SHAPE[:2]
    _assert_image_untouched(scene)


@pytest.mark.parametrize("leave", ["delete labels", "select image"])
def test_image_is_untouched_when_the_labels_layer_is_left(viewer, scene, leave):
    # the blur is applied to the plane given to the calculator only, never to the image layer
    mc = scene.mc
    mc.blur_image_checkbox.value = True
    viewer.dims.set_point(0, 2)
    _enable_with_button(viewer, scene.labels)
    _assert_image_untouched(scene)

    if leave == "delete labels":
        viewer.layers.remove(scene.labels)
    else:
        viewer.layers.selection.active = scene.image
    _process_events()
    assert mc.labels_layer is None
    _assert_image_untouched(scene)

    if leave == "select image":
        # back on the labels layer, whose tool is still on
        viewer.layers.selection.active = scene.labels
        _process_events()
        assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]
        _assert_image_untouched(scene)


def test_tool_state_follows_the_active_labels_layer(viewer, scene, feature_calls):
    other = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint8), name="other",
                              metadata={"source_image_layer": scene.image})
    viewer.layers.selection.active = scene.labels
    _process_events()
    scene.mc._enable_mc_mode()
    _process_events()
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]

    viewer.layers.selection.active = other
    _process_events()
    assert not other._overlays["minimal_contour"].enabled
    assert VispyMinimalContourOverlay._features_shape is None

    n_calls = len(feature_calls)
    viewer.layers.selection.active = scene.labels
    _process_events()
    assert len(feature_calls) > n_calls
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]


@pytest.mark.parametrize("mc_on", [False, True])
def test_blur_preview_follows_the_sigma_slider(viewer, scene, mc_on):
    # the preview used Image._set_view_slice(), which raises NotImplementedError on newer napari
    mc = scene.mc
    viewer.dims.set_point(0, 2)
    if mc_on:
        mc._enable_mc_mode()
    _process_events()
    mc.blur_image_checkbox.value = True
    previews = []
    for sigma in (0.5, 2.5):
        mc.blur_sigma_slider.value = sigma
        previews.append(np.asarray(mc.demo_image_widget.value).copy())
    assert previews[0].shape[-1] == 3
    assert not np.array_equal(*previews)


def test_enabled_callback_is_connected_once(viewer, scene, feature_calls):
    # re-selecting the layer must not connect the callback twice: the features would be prepared twice
    viewer.layers.selection.events.active(value=scene.labels)
    _process_events()
    scene.mc._enable_mc_mode()
    _process_events()
    assert feature_calls == [id(scene.image)]
