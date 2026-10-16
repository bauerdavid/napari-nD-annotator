"""End-to-end test of the Minimal Surface widget.

A real (never shown) ``napari.Viewer`` and ``AnnotatorWidget`` are created, a synthetic ball and
two points on its equator are added and the widget's own ``_start_estimation`` is called. The user
pressing "Done" on the slice annotation is simulated, and every exception raised on the way
(main thread, Qt slots, worker thread) is collected.

The test needs the optional ``minimal_surface`` package and an OpenGL context (napari creates its
canvas even if the viewer is not shown), so it is skipped when either is missing. On Linux CI run it
under xvfb with software GL, with ``QT_QPA_PLATFORM`` unset (the ``offscreen`` platform has no GL).
Set ``REQUIRE_MINIMAL_SURFACE=1`` (as the CI jobs that install minimal-surface do) to make a missing
package or a missing GL context a failure instead of a skip.

Besides the exceptions, the test fails if the run triggered a deprecation warning from the plugin
(the ``deprecations`` fixture of ``test_napari_compat.py``).
"""
import os
import threading
import time
import traceback

import numpy as np
import pytest
from qtpy import QtWidgets

# the fixtures of the compatibility tests: the Annotation Toolbox docked in a viewer, and the deprecation check
from .test_napari_compat import deprecations, annotator  # noqa: F401


def _skip(reason):
    if os.environ.get("REQUIRE_MINIMAL_SURFACE") == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason, allow_module_level=True)


try:
    import minimal_surface  # noqa: F401
except ImportError as e:
    _skip("the optional minimal_surface package is not installed: %r" % e)

BALL_CENTER = (20, 30, 30)
BALL_SHAPE = (40, 60, 60)
TIMEOUT_MS = 90_000


def _ball_image(dtype):
    zz, yy, xx = np.mgrid[tuple(slice(0, s) for s in BALL_SHAPE)]
    dist = np.sqrt((zz - BALL_CENTER[0]) ** 2 + (yy - BALL_CENTER[1]) ** 2 + (xx - BALL_CENTER[2]) ** 2)
    ball = np.exp(-((dist / 14.0) ** 6))
    noise = np.random.default_rng(0).normal(0, 20, BALL_SHAPE)
    return (ball * 1000 + noise).clip(0, None).astype(dtype)


@pytest.fixture
def viewer(qtbot):
    import napari
    try:
        viewer = napari.Viewer(show=False)
    except Exception as e:  # no GL context / no usable Qt platform
        _skip("cannot create a napari viewer (no OpenGL context?): %r" % e)
    yield viewer
    viewer.close()
    QtWidgets.QApplication.processEvents()


def _thread_running(worker):
    try:
        thread = worker.thread()
    except RuntimeError:  # the worker was deleted together with its finished thread
        return False
    return thread is not QtWidgets.QApplication.instance().thread() and thread.isRunning()


@pytest.fixture
def workers(monkeypatch):
    """Every EstimationWorker created during the test."""
    from napari_nd_annotator._widgets import minimal_surface_widget as msw
    created = []
    original_init = msw.EstimationWorker.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(msw.EstimationWorker, "__init__", init)
    yield created
    # a worker still waiting for "Done" would keep running while the viewer is torn down
    for worker in created:
        worker.stop_requested = True
        worker.done_pressed = True
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and any(_thread_running(w) for w in created):
        QtWidgets.QApplication.processEvents()
        time.sleep(0.02)


@pytest.fixture(autouse=True)
def no_dialogs(monkeypatch):
    """Never block on (or show) a modal dialog; answer "Yes"/OK."""
    monkeypatch.setattr(QtWidgets.QDialog, "show", lambda self, *a, **k: None)
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self, *a, **k: QtWidgets.QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))


# F1-F4 of the 2026-10-12 widget scan: the "Minimal contour" slice annotation, z_scale, float32 images (fixed in
# minimal-surface master) and a Labels layer of another shape; they were strict xfails until fixed
@pytest.mark.parametrize(
    "dtype, method, z_scale, mismatch",
    [
        ("float64", "Threshold", 1.0, False),
        ("uint16", "Threshold", 1.0, False),
        ("float64", "Manual", 1.0, False),
        ("uint16", "Manual", 1.0, False),
        pytest.param("float64", "Minimal contour", 1.0, False, id="float64-Minimal contour-F1"),
        pytest.param("float64", "Threshold", 2.0, False, id="float64-Threshold-F2"),
        pytest.param("float32", "Threshold", 1.0, False, id="float32-Threshold-F3"),
        pytest.param("float64", "Threshold", 1.0, True, id="float64-Threshold-F4"),
    ],
)
def test_minimal_surface_estimation(qtbot, monkeypatch, viewer, annotator, workers, dtype, method, z_scale, mismatch):
    with qtbot.capture_exceptions() as exceptions:
        monkeypatch.setattr(threading, "excepthook",
                            lambda args: exceptions.append((args.exc_type, args.exc_value, args.exc_traceback)))

        ms = annotator.ToolsWidget.minimal_surface_widget

        layer = viewer.add_image(_ball_image(dtype), name="ball")
        ms.image_layer = layer
        qtbot.waitUntil(lambda: ms.image_layer is layer, timeout=5000)
        ms.slice_annotation_method = method
        ms.z_scale = z_scale
        if mismatch:
            viewer.dims.set_point(0, 5)  # a slice that the small labels layer also has
            ms.labels_layer = viewer.add_labels(np.zeros((10, 10, 10), dtype=np.uint16), name="wrong shape")
        ms.points_layer.add(np.array([[20., 30., 16.], [20., 30., 44.]]))

        state = {"all_done": False, "annotated": False}
        ms._start_estimation()

        def step():
            """Simulate the user; return True when the estimation has finished (or failed)."""
            if exceptions:
                return True
            if not workers:
                return False
            worker = workers[0]
            if not getattr(worker, "_test_hooked", False):
                worker.all_done.connect(lambda: state.update(all_done=True))
                worker._test_hooked = True
            if (not state["annotated"] and worker.slice_labels_layer is not None
                    and worker.slice_image_layer is not None):
                state["annotated"] = True
                data = worker.slice_image_layer.data
                worker.slice_labels_layer.data = (data >= 0.5 * data.max()).astype(int)
                worker.finish_slice_annotation()
                worker.done_pressed = True
            return state["all_done"]

        try:
            qtbot.waitUntil(step, timeout=TIMEOUT_MS)
        except TimeoutError:
            pass  # reported by the assertions below
        # let the queued signals (mask_data_received, thread quit) be delivered
        qtbot.wait(200)

    assert not exceptions, "exceptions raised:\n" + "\n".join(
        "".join(traceback.format_exception(*e)) for e in exceptions)
    assert state["annotated"], "the slice annotation never started"
    assert state["all_done"], "the estimation did not finish"

    results = [np.asarray(l.data) > 0 for l in viewer.layers
               if l.__class__.__name__ == "Labels" and l.data.shape == BALL_SHAPE]
    assert results, "no result Labels layer"
    mask = max(results, key=np.count_nonzero)
    voxels = np.count_nonzero(mask)
    assert voxels > 0, "the result Labels layer is empty"

    # roughly a ball around the centre: size close to the 4139 voxels seen on a working setup, centred, compact
    assert 2000 < voxels < 10000, voxels
    coords = np.argwhere(mask)
    assert np.allclose(coords.mean(axis=0), BALL_CENTER, atol=3), coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0) + 1
    assert extent.max() / extent.min() < 1.5, extent
    assert voxels / np.prod(extent) > 0.15, (voxels, extent)  # not a thin shell or a few stray voxels
