"""Smoke tests that exercise the plugin against whichever napari is installed.

They need a working OpenGL context (the viewer is created but never shown), so on a
headless machine run them under ``xvfb-run`` with a software GL driver. The CI job
``napari_compat.yml`` runs them for several napari versions.
"""
import sys
import threading
import time
import traceback
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from napari_nd_annotator._helper_functions import viewer_camera, viewer_qt_viewer


def _process_events():
    from qtpy.QtWidgets import QApplication
    QApplication.processEvents()


def _plugin_deprecations(caught):
    """Deprecation warnings raised while the plugin ran, apart from pydantic's own noise."""
    return [
        w for w in caught
        if issubclass(w.category, (DeprecationWarning, FutureWarning, PendingDeprecationWarning))
        and not w.category.__name__.startswith("PydanticDeprecated")
        and "pydantic" not in str(w.message).lower()
    ]


# --- helpers, independent of the installed napari --------------------------------------

def test_viewer_camera_prefers_scene_camera():
    viewer = SimpleNamespace(scene=SimpleNamespace(camera="scene camera"), camera="legacy camera")
    assert viewer_camera(viewer) == "scene camera"


def test_viewer_camera_falls_back_to_legacy_camera():
    assert viewer_camera(SimpleNamespace(camera="legacy camera")) == "legacy camera"


def test_viewer_qt_viewer_prefers_private_attribute():
    viewer = SimpleNamespace(window=SimpleNamespace(_qt_viewer="private", qt_viewer="public"))
    assert viewer_qt_viewer(viewer) == "private"


def test_viewer_qt_viewer_falls_back_to_public_attribute():
    assert viewer_qt_viewer(SimpleNamespace(window=SimpleNamespace(qt_viewer="public"))) == "public"


# --- the plugin inside a real viewer ---------------------------------------------------

@pytest.fixture
def viewer():
    import napari
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()
    _process_events()


@pytest.fixture
def deprecations(viewer):
    """Records the warnings raised from here on; the test fails if the plugin triggered a deprecation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught
        _process_events()
    found = _plugin_deprecations(caught)
    assert not found, "deprecated API used:\n" + "\n".join(
        "%s: %s (%s:%s)" % (w.category.__name__, w.message, w.filename, w.lineno) for w in found
    )


@pytest.fixture
def annotator(viewer, deprecations):
    from napari_nd_annotator import AnnotatorWidget
    widget = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(widget, name="Annotator")
    yield widget


def test_viewer_helpers_on_installed_napari(viewer):
    assert viewer_camera(viewer).zoom > 0
    assert hasattr(viewer_qt_viewer(viewer), "controls")
    from napari_nd_annotator._helper_functions import viewer_canvas_size
    assert len(viewer_canvas_size(viewer)) == 2


def test_labels_layer_gets_overlays_and_controls(viewer, annotator):
    viewer.add_image(np.zeros((5, 20, 20)), name="image")
    labels = viewer.add_labels(np.zeros((5, 20, 20), dtype=np.uint16), name="labels")
    viewer.layers.selection.active = labels
    _process_events()
    assert "interpolation" in labels._overlays
    assert "minimal_contour" in labels._overlays
    assert labels in viewer_qt_viewer(viewer).controls.widgets


def test_slice_annotation_fits_and_restores_camera(viewer, annotator):
    from napari_nd_annotator._widgets.minimal_surface_widget import EstimationWorker
    tools = annotator.ToolsWidget
    # without the minimal-surface package the Minimal Surface widget is None; the worker only needs its call button
    surface_widget = tools.minimal_surface_widget or SimpleNamespace(call_button=SimpleNamespace(enabled=False))
    worker = EstimationWorker(viewer, tools.minimal_contour_widget, surface_widget)
    worker.init_slice_annotation_fun = lambda: None
    worker.finish_slice_annotation_fun = lambda: None
    viewer.add_image(np.zeros((5, 20, 20)), name="image")
    camera = viewer_camera(viewer)
    zoom_before = camera.zoom

    worker.begin_slice_annotation(np.zeros((30, 40)), None)
    assert viewer.dims.ndisplay == 2
    assert worker.slice_labels_layer in viewer.layers
    assert camera.zoom != zoom_before

    worker.finish_slice_annotation()
    assert worker.slice_labels_layer is None
    assert camera.zoom == pytest.approx(zoom_before)


# --- end to end, needs the optional minimal-surface package -----------------------------

def test_minimal_surface_estimation(viewer, annotator):
    """The scenario of the Minimal Surface widget: two points, threshold annotation, 3D result."""
    pytest.importorskip("minimal_surface")
    from qtpy import QtWidgets
    from napari_nd_annotator._widgets import minimal_surface_widget as msw

    errors = []
    workers = []
    original_init = msw.EstimationWorker.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        workers.append(self)

    ms = annotator.ToolsWidget.minimal_surface_widget
    shape = (40, 60, 60)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    ball = np.exp(-((np.sqrt((zz - 20) ** 2 + (yy - 30) ** 2 + (xx - 30) ** 2) / 14.0) ** 6))
    image = (ball * 1000 + np.random.default_rng(0).normal(0, 20, shape)).clip(0, None)

    # no dialog may block the test or pop up on the desktop: questions are answered "Yes", modal dialogs OK
    yes = QtWidgets.QMessageBox.StandardButton.Yes
    with mock.patch.object(msw.EstimationWorker, "__init__", init), \
            mock.patch.object(QtWidgets.QWidget, "show", lambda self, *a, **k: None), \
            mock.patch.object(QtWidgets.QDialog, "exec", lambda self, *a, **k: QtWidgets.QDialog.DialogCode.Accepted), \
            mock.patch.object(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: yes)), \
            mock.patch.object(sys, "excepthook", lambda *exc: errors.append("".join(traceback.format_exception(*exc)))), \
            mock.patch.object(threading, "excepthook", lambda a: errors.append(
                "".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))):
        layer = viewer.add_image(image.astype(np.float64), name="ball")
        ms.image_layer = layer
        _process_events()
        ms.slice_annotation_method = "Threshold"
        ms.z_scale = 1.0
        ms.points_layer.add(np.array([[20., 30., 16.], [20., 30., 44.]]))
        ms._start_estimation()

        done = []
        handled = False
        deadline = time.time() + 120
        while time.time() < deadline and not done:
            _process_events()
            if workers:
                worker = workers[0]
                if not getattr(worker, "_hooked", False):
                    worker.all_done.connect(lambda: done.append(True))
                    worker._hooked = True
                # act as the user: threshold annotation is shown, accept it
                if not handled and worker.slice_labels_layer is not None and worker.slice_image_layer is not None:
                    handled = True
                    data = worker.slice_image_layer.data
                    worker.slice_labels_layer.data = (data >= 0.5 * data.max()).astype(int)
                    worker.finish_slice_annotation()
                    worker.done_pressed = True
            time.sleep(0.02)

    assert not errors, "\n".join(errors)
    assert done, "the estimation did not finish"
    result = [l for l in viewer.layers if l.__class__.__name__ == "Labels"]
    assert result
    voxels = int((np.asarray(result[0].data) > 0).sum())
    assert 3500 < voxels < 4800, voxels   # 4139 when written
