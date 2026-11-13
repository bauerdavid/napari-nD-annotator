"""End-to-end test of the Minimal Surface widget.

A real (never shown) ``napari.Viewer`` and ``AnnotatorWidget`` are created, a synthetic ball with an object (a
bounding box with a label id) around it and two anchors on its equator are added and the widget's own
``_start_estimation`` is called. The user pressing "Done" on the slice annotation is simulated, and every exception
raised on the way (main thread, Qt slots, worker thread) is collected. The result has to be written with the label
id of the object, and Ctrl+Z has to take it back.

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
import warnings

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

try:
    import napari_bbox  # noqa: F401
except ImportError as e:
    _skip("the napari-bbox plugin (the object list) is not installed: %r" % e)

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


OBJECT_LABEL = 5


def _add_object(viewer, image_layer, labels_layer, label=OBJECT_LABEL):
    """An object list with one object (the box is the whole image) that is selected."""
    import itertools
    from napari_bbox import BoundingBoxLayer
    from napari_nd_annotator._widgets.object_list import ObjectListWidget
    low, high = np.zeros(3), np.asarray(BALL_SHAPE, dtype=float) - 1
    corners = np.where(list(itertools.product((False, True), repeat=3)), high, low)
    bbox_layer = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(bbox_layer)
    bbox_layer.data = [corners]
    bbox_layer.features = {"label": np.array([label])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # the object list sets the text of the boxes the old way
        object_list = ObjectListWidget(viewer, "Object", bbox_layer, image_layer, labels_layer, None, label + 1)
        QtWidgets.QApplication.processEvents()
        object_list.select_item(object_list.item(0))
        QtWidgets.QApplication.processEvents()
    return bbox_layer, object_list


# F1-F3 of the 2026-10-12 widget scan: the "Minimal contour" slice annotation, z_scale, float32 images (fixed in
# minimal-surface master); they were strict xfails until fixed
@pytest.mark.parametrize(
    "dtype, method, z_scale",
    [
        ("float64", "Threshold", 1.0),
        ("uint16", "Threshold", 1.0),
        ("float64", "Manual", 1.0),
        ("uint16", "Manual", 1.0),
        pytest.param("float64", "Minimal contour", 1.0, id="float64-Minimal contour-F1"),
        pytest.param("float64", "Threshold", 2.0, id="float64-Threshold-F2"),
        pytest.param("float32", "Threshold", 1.0, id="float32-Threshold-F3"),
        pytest.param("float64", "Automatic (distances)", 1.0, id="float64-Automatic-distances"),
        pytest.param("uint16", "Automatic (distances)", 2.0, id="uint16-Automatic-distances-z2"),
        pytest.param("float64", "Automatic (phase field)", 1.0, id="float64-Automatic-phase-field"),
    ],
)
def test_minimal_surface_estimation(qtbot, monkeypatch, viewer, annotator, workers, dtype, method, z_scale):
    with qtbot.capture_exceptions() as exceptions:
        monkeypatch.setattr(threading, "excepthook",
                            lambda args: exceptions.append((args.exc_type, args.exc_value, args.exc_traceback)))

        ms = annotator.ToolsWidget.minimal_surface_widget

        layer = viewer.add_image(_ball_image(dtype), name="ball")
        ms.image_layer = layer
        qtbot.waitUntil(lambda: ms.image_layer is layer, timeout=5000)
        from napari_nd_annotator._widgets import minimal_surface_widget as msw
        automatic = method in msw.EstimationWorker.AUTOMATIC_METHODS
        if method not in msw.annotation_methods():
            pytest.skip("this minimal_surface build has no automatic slice annotation")
        ms.slice_annotation_method = method
        ms.z_scale = z_scale
        labels = viewer.add_labels(np.zeros(BALL_SHAPE, dtype=np.uint16), name="labels")
        # the object was segmented before, and another object is next to it
        labels.data[0:3, 0:3, 0:3] = OBJECT_LABEL
        labels.data[0:3, 58:60, 58:60] = OBJECT_LABEL + 1
        bbox_layer, object_list = _add_object(viewer, layer, labels)
        from napari_nd_annotator.object_anchors import anchor_store
        anchor_store(bbox_layer).set(OBJECT_LABEL, [[20., 30., 16.], [20., 30., 44.]])

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
    # the automatic methods need no annotation: no dialog or temporary layers, the view stays as it is
    assert state["annotated"] != automatic, "the slice annotation %s" % ("started" if automatic else "never started")
    assert state["all_done"], "the estimation did not finish"

    # the result is the object, with its label id: the previous mask of the object is gone, the other object stays
    assert set(np.unique(labels.data)) == {0, OBJECT_LABEL, OBJECT_LABEL + 1}
    assert np.all(labels.data[0:3, 0:3, 0:3] == 0) and np.all(labels.data[0:3, 58:60, 58:60] == OBJECT_LABEL + 1)
    assert viewer.layers.selection.active is labels  # Ctrl+Z works
    mask = np.asarray(labels.data) == OBJECT_LABEL
    voxels = np.count_nonzero(mask)
    assert voxels > 0, "the result Labels layer is empty"

    # roughly a ball around the centre, centred, compact. The annotated methods give a smaller blob (4139 voxels seen
    # on a working setup); the automatic ones find the ball (radius 14, 11494 voxels) from the anchors alone
    low, high = (8000, 13000) if automatic else (2000, 10000)
    assert low < voxels < high, voxels
    coords = np.argwhere(mask)
    assert np.allclose(coords.mean(axis=0), BALL_CENTER, atol=3), coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0) + 1
    assert extent.max() / extent.min() < 1.5, extent
    assert voxels / np.prod(extent) > 0.15, (voxels, extent)  # not a thin shell or a few stray voxels

    # Ctrl+Z takes the result back (one step), with the previous state of the object
    labels.undo()
    assert not np.any(labels.data == OBJECT_LABEL) or np.count_nonzero(labels.data == OBJECT_LABEL) == 27
    assert np.all(labels.data[0:3, 0:3, 0:3] == OBJECT_LABEL)
    assert np.all(labels.data[0:3, 58:60, 58:60] == OBJECT_LABEL + 1)
