"""The Minimal Contour widget only processes the source image while the minimal contour tool is on.

Painting and scrolling with the other labels tools must not copy the image, compute image features or
write feature files: on large volumes that work froze napari on every slice change.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers.labels._labels_constants import Mode

from napari_nd_annotator._helper_functions import viewer_qt_viewer
from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
from napari_nd_annotator.minimal_contour.feature_manager import FeatureManager

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
    """The layers FeatureManager.get_features was asked for."""
    calls = []
    get_features = FeatureManager.get_features

    def spy(self, layer, *args, **kwargs):
        calls.append(layer)
        return get_features(self, layer, *args, **kwargs)
    monkeypatch.setattr(FeatureManager, "get_features", spy)
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
    return SimpleNamespace(mc=annotator.ToolsWidget.minimal_contour_widget, data=data, image=image, labels=labels)


def _temp_files(mc):
    folder = mc.feature_manager.temp_folder
    return os.listdir(folder) if os.path.exists(folder) else []


def _assert_idle(scene, feature_calls):
    assert feature_calls == []
    assert scene.image.data is scene.data
    assert scene.mc._orig_image is None
    assert _temp_files(scene.mc) == []
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
    assert scene.image in feature_calls
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]
    assert _temp_files(scene.mc) != []


def test_turning_the_tool_off_leaves_nothing_behind(viewer, scene, feature_calls):
    mc = scene.mc
    mc.blur_image_checkbox.value = True
    viewer.dims.set_point(0, 2)
    _enable_with_button(viewer, scene.labels)
    # the blurred slice is shown in the image layer while the tool is on
    assert not np.array_equal(scene.image.data[2], scene.data[2])

    scene.labels.mode = Mode.PAINT
    _process_events()
    assert not scene.labels._overlays["minimal_contour"].enabled
    assert VispyMinimalContourOverlay._features_shape is None
    np.testing.assert_array_equal(scene.image.data, scene.data)

    n_calls = len(feature_calls)
    _scroll(viewer, mc, [4, 5])
    assert len(feature_calls) == n_calls
    np.testing.assert_array_equal(scene.image.data, scene.data)

    # on again, on another slice and with a transposed display order
    mc._enable_mc_mode()
    _process_events()
    assert len(feature_calls) > n_calls
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]
    assert not np.array_equal(scene.image.data[5], scene.data[5])
    viewer.dims.order = (2, 0, 1)
    _process_events()
    assert VispyMinimalContourOverlay._features_shape == SHAPE[:2]


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


def test_enabled_callback_is_connected_once(viewer, scene, feature_calls):
    # re-selecting the layer must not connect the callback twice: the features would be prepared twice
    viewer.layers.selection.events.active(value=scene.labels)
    _process_events()
    scene.mc._enable_mc_mode()
    _process_events()
    assert feature_calls == [scene.image]
