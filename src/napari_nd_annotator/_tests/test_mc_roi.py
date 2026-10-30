"""The Minimal Contour widget gives the calculator a window (ROI) of the displayed plane instead of the whole plane.

Two sources: the bounding box of the selected object (while the contour runs on that very image) and, for huge
planes, a window around the anchors. Inside the window the contour must equal the one of the whole plane.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from napari_nd_annotator._widgets import minimal_contour_widget as mcw
from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
from napari_nd_annotator._widgets.minimal_contour_widget import (
    ASSYM_GRADIENT, CUSTOM, HIGH_INTENSITY, LOW_INTENSITY, SYM_GRADIENT)
from napari_nd_annotator.minimal_contour import roi as mc_roi
from napari_nd_annotator.minimal_contour.roi import Box  # noqa: F401

SHAPE = (3, 300, 360)
BBOX_MIN = np.array([0, 100, 120])
BBOX_MAX = np.array([2, 160, 200])
# anchors (rows, columns) of a contour around the bright disc, all inside the bounding box (the calculator has 3)
ANCHORS = [[101., 160.], [145., 190.], [150., 130.]]
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


@pytest.fixture(autouse=True)
def reset_calculator():
    yield
    VispyMinimalContourOverlay.set_calculator_feature(None)


@pytest.fixture
def calls(monkeypatch):
    """The arguments of every set_calculator_feature call (arrays copied) and the shape of the plane it was for."""
    recorded = []
    original = VispyMinimalContourOverlay.set_calculator_feature

    def spy(features, grad_x=None, grad_y=None, **kwargs):
        recorded.append(SimpleNamespace(
            arrays=tuple(None if a is None else np.array(a) for a in (features, grad_x, grad_y)), kwargs=kwargs))
        return original(features, grad_x, grad_y, **kwargs)
    monkeypatch.setattr(VispyMinimalContourOverlay, "set_calculator_feature", staticmethod(spy))
    return recorded


def _blobs(shape, rgb=False, seed=0):
    """Noisy planes (``shape`` ends with rows, columns) with a bright disc where the anchors are and a second one."""
    yy, xx = np.mgrid[:shape[-2], :shape[-1]]
    rng = np.random.default_rng(seed)
    disc = ((yy - 130) ** 2 + (xx - 160) ** 2 < 35 ** 2) | ((yy - 40) ** 2 + (xx - 300) ** 2 < 25 ** 2)
    plane = 700 * disc + rng.random(shape) * 120
    if rgb:
        plane = np.stack([plane, np.roll(plane, 5, axis=-1) * 0.8, plane[..., ::-1] * 0.6], axis=-1)
    return plane.astype(np.uint16)


def _scene(viewer, data, feature=ASSYM_GRADIENT, blur=None, **image_kwargs):
    from napari_nd_annotator import AnnotatorWidget
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    image = viewer.add_image(data, name="image", **image_kwargs)
    labels = viewer.add_labels(np.zeros(data.shape[:image.ndim], dtype=np.uint16), name="labels",
                               metadata={"source_image_layer": image})
    viewer.layers.selection.active = labels
    image.contrast_limits = LIMITS
    mc = annotator.ToolsWidget.minimal_contour_widget
    mc.used_feature_combobox.value = feature
    mc.blur_image_checkbox.value = blur is not None
    if blur is not None:
        mc.blur_sigma_slider.value = blur
    mc._enable_mc_mode()
    _process_events()
    return SimpleNamespace(annotator=annotator, mc=mc, image=image, labels=labels)


def _select(monkeypatch, scene, bbox_min=BBOX_MIN, bbox_max=BBOX_MAX, image=None):
    """Pretend that an object is selected in the object list (which needs napari_bbox)."""
    selected = (4, np.array(bbox_min), np.array(bbox_max), scene.image if image is None else image)
    monkeypatch.setattr(mcw, "selected_object", lambda viewer: selected)


def _deselect(monkeypatch):
    monkeypatch.setattr(mcw, "selected_object", lambda viewer: None)


def _settle(scene):
    _process_events()
    scene.mc._set_image()
    _process_events()


def _estimate(anchors):
    """What the overlay's mouse move computes for ``anchors`` (rows, columns), without a viewer."""
    overlay = VispyMinimalContourOverlay.__new__(VispyMinimalContourOverlay)
    overlay.point_triangle = np.asarray(anchors, dtype=float)
    return overlay._estimate()


def _assert_same_contours(actual, expected):
    assert actual is not None and len(actual) == len(expected) and sum(len(s) for s in expected) > 50
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def _expected_box(plane_shape=SHAPE[1:], dims=(1, 2)):
    return mc_roi.object_box(BBOX_MIN, BBOX_MAX, dims, plane_shape)


MODES = [
    pytest.param(dict(feature=ASSYM_GRADIENT), False, id="assym"),
    pytest.param(dict(feature=ASSYM_GRADIENT, blur=2.0), False, id="assym-blur"),
    pytest.param(dict(feature=HIGH_INTENSITY), False, id="high-intensity"),
    pytest.param(dict(feature=LOW_INTENSITY, blur=1.0), False, id="low-intensity-blur"),
    pytest.param(dict(feature=ASSYM_GRADIENT), True, id="assym-rgb"),
    pytest.param(dict(feature=ASSYM_GRADIENT, blur=1.5), True, id="assym-rgb-blur"),
]


@pytest.mark.parametrize("options, rgb", MODES)
def test_object_roi_gives_the_whole_plane_contour(viewer, monkeypatch, calls, options, rgb):
    scene = _scene(viewer, _blobs(SHAPE, rgb=rgb), **options)
    viewer.dims.set_point(0, 1)
    _deselect(monkeypatch)
    _settle(scene)
    whole = calls[-1]
    assert whole.kwargs == {} and VispyMinimalContourOverlay._roi_origin == (0, 0)
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]
    expected = _estimate(ANCHORS)

    _select(monkeypatch, scene)
    _settle(scene)
    box = _expected_box()
    assert box is not None and box.area < 0.6 * SHAPE[1] * SHAPE[2]
    assert VispyMinimalContourOverlay._features_shape == box.shape
    assert VispyMinimalContourOverlay._roi_origin == box.origin
    assert VispyMinimalContourOverlay._plane_shape == SHAPE[1:]
    window = calls[-1]
    assert window.kwargs == dict(roi_origin=box.origin, plane_shape=SHAPE[1:])
    # what the calculator got is the window of what it gets without a ROI
    for part, full in zip(window.arrays, whole.arrays):
        if part is not None:
            np.testing.assert_array_equal(part, full[box.slices])
    _assert_same_contours(_estimate(ANCHORS), expected)


def test_symmetric_gradient_roi_is_normalised_with_its_own_range(viewer, monkeypatch, calls):
    scene = _scene(viewer, _blobs(SHAPE), feature=SYM_GRADIENT)
    viewer.dims.set_point(0, 1)
    _deselect(monkeypatch)
    _settle(scene)
    whole = calls[-1].arrays[0][..., 0] if calls[-1].arrays[0].ndim == 3 else calls[-1].arrays[0]
    _select(monkeypatch, scene)
    _settle(scene)
    box = _expected_box()
    part = calls[-1].arrays[0][..., 0] if calls[-1].arrays[0].ndim == 3 else calls[-1].arrays[0]
    assert part.shape == box.shape and VispyMinimalContourOverlay._roi_origin == box.origin
    assert part.min() == 0 and part.max() == 1  # scaled to the window's own range
    # the same map up to that scaling
    reference = whole[box.slices]
    scale = (reference.max() - reference.min()) / (part.max() - part.min())
    np.testing.assert_allclose(part * scale + reference.min(), reference, atol=0.15 * reference.max())
    contour = _estimate(ANCHORS)
    assert contour is not None and sum(len(s) for s in contour) > 50


def test_no_contrast_limits_normalise_with_the_roi_range(viewer, monkeypatch, calls):
    scene = _scene(viewer, _blobs(SHAPE), feature=HIGH_INTENSITY)
    scene.mc.apply_contrast_limits = False
    viewer.dims.set_point(0, 1)
    _select(monkeypatch, scene)
    _settle(scene)
    box = _expected_box()
    plane = scene.image.data[1][box.slices].astype(float)
    expected = (plane - plane.min()) / (plane.max() - plane.min())
    np.testing.assert_allclose(calls[-1].arrays[0][..., 0], expected)


def test_object_roi_in_a_transposed_display_order(viewer, monkeypatch, calls):
    scene = _scene(viewer, np.ascontiguousarray(_blobs(SHAPE).transpose(1, 0, 2)))
    data = scene.image.data  # (300, 3, 360): rows, slices, columns
    assert data.shape == (300, 3, 360)
    viewer.dims.order = (1, 0, 2)  # the plane shows (rows, columns) as (axis 0, axis 2), slices along axis 1
    viewer.dims.set_point(1, 1)
    _deselect(monkeypatch)
    _settle(scene)
    expected = _estimate(ANCHORS)
    whole = calls[-1].arrays

    bbox_min, bbox_max = np.array([100, 0, 120]), np.array([160, 2, 200])
    _select(monkeypatch, scene, bbox_min, bbox_max)
    _settle(scene)
    dims = tuple(mcw.layer_dims_displayed(scene.image))
    assert dims == (0, 2)
    box = mc_roi.object_box(bbox_min, bbox_max, dims, (300, 360))
    assert VispyMinimalContourOverlay._roi_origin == box.origin
    np.testing.assert_array_equal(calls[-1].arrays[0], whole[0][box.slices])
    _assert_same_contours(_estimate(ANCHORS), expected)

    # transposed: the plane shows axis 2 along the rows
    viewer.dims.order = (1, 2, 0)
    _settle(scene)
    dims = tuple(mcw.layer_dims_displayed(scene.image))
    assert dims == (2, 0)
    box = mc_roi.object_box(bbox_min, bbox_max, dims, (360, 300))
    assert box is not None
    assert VispyMinimalContourOverlay._features_shape == box.shape
    assert VispyMinimalContourOverlay._roi_origin == box.origin
    assert VispyMinimalContourOverlay._plane_shape == (360, 300)


def test_leaving_the_object_box_grows_the_roi(viewer, monkeypatch, calls):
    scene = _scene(viewer, _blobs(SHAPE))
    viewer.dims.set_point(0, 1)
    _deselect(monkeypatch)
    _settle(scene)
    outside = [[101., 160.], [145., 190.], [230., 130.]]  # reaches far below the object's box
    expected = _estimate(outside)

    _select(monkeypatch, scene)
    _settle(scene)
    first_box = _expected_box()
    assert VispyMinimalContourOverlay._roi_origin == first_box.origin
    n_calls = len(calls)
    actual = _estimate(outside)
    assert len(calls) == n_calls + 1  # one request, answered by a bigger window
    assert VispyMinimalContourOverlay._roi_origin[0] <= first_box.y0
    assert VispyMinimalContourOverlay._features_shape[0] > first_box.shape[0] or VispyMinimalContourOverlay._plane_shape is None
    _assert_same_contours(actual, expected)
    # asking again does not need another window
    _estimate(outside)
    assert len(calls) == n_calls + 1

    # the next settled plane starts again from the object's box
    _settle(scene)
    assert VispyMinimalContourOverlay._roi_origin == first_box.origin


@pytest.mark.parametrize("case", ["other-image", "big-box", "outside-plane", "custom", "error", "off"])
def test_the_whole_plane_is_used_when_there_is_no_usable_roi(viewer, monkeypatch, calls, case):
    scene = _scene(viewer, _blobs(SHAPE))
    viewer.dims.set_point(0, 1)
    if case == "other-image":
        other = viewer.add_image(np.zeros(SHAPE), name="other")
        viewer.layers.selection.active = scene.labels
        _select(monkeypatch, scene, image=other)
    elif case == "big-box":
        _select(monkeypatch, scene, [0, 5, 5], [2, 290, 350])
    elif case == "outside-plane":
        _select(monkeypatch, scene, [0, 1000, 1000], [2, 1100, 1100])
    elif case == "custom":
        _select(monkeypatch, scene)
        monkeypatch.setattr(type(scene.mc), "used_feature", property(lambda self: CUSTOM))
        assert scene.mc._choose_roi(scene.image, SHAPE[1:]) is None
        return
    elif case == "error":
        def broken(viewer):
            raise RuntimeError("no object list")
        monkeypatch.setattr(mcw, "selected_object", broken)
    else:
        _deselect(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _settle(scene)
    assert calls[-1].kwargs == {}
    assert VispyMinimalContourOverlay._features_shape == SHAPE[1:]
    assert VispyMinimalContourOverlay._roi_origin == (0, 0) and VispyMinimalContourOverlay._plane_shape is None


def test_nothing_is_computed_in_3d_display(viewer, monkeypatch, calls):
    scene = _scene(viewer, _blobs(SHAPE))
    _select(monkeypatch, scene)
    viewer.dims.set_point(0, 1)
    _settle(scene)
    assert VispyMinimalContourOverlay._plane_shape == SHAPE[1:]
    n_features = len([c for c in calls if c.arrays[0] is not None])
    viewer.dims.ndisplay = 3
    _settle(scene)
    assert len([c for c in calls if c.arrays[0] is not None]) == n_features  # nothing is computed in 3D


def test_no_roi_for_multiscale_images(viewer, monkeypatch):
    from napari_nd_annotator import AnnotatorWidget
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    data = _blobs(SHAPE)
    image = viewer.add_image([data, data[:, ::2, ::2]], multiscale=True, name="multi")
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels", metadata={"source_image_layer": image})
    viewer.layers.selection.active = labels
    mc = annotator.ToolsWidget.minimal_contour_widget
    monkeypatch.setattr(mcw, "selected_object", lambda viewer: (1, BBOX_MIN, BBOX_MAX, image))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc._enable_mc_mode()
        _process_events()
    assert VispyMinimalContourOverlay._features_shape is None and VispyMinimalContourOverlay._plane_shape is None


# ---- the window around the anchors of a huge plane ------------------------------------------------------------------

@pytest.fixture
def small_windows(monkeypatch):
    """Treat these planes as huge: window them from 1000 pixels on, with windows of at least 80 x 80."""
    monkeypatch.setattr(mc_roi, "LARGE_PLANE_PIXELS", 1000)
    monkeypatch.setattr(mc_roi, "WINDOW_SIZE", 80)


def _set_anchors(scene, anchors, z=1):
    scene.labels._overlays["minimal_contour"].anchor_points = [[z, y, x] for y, x in anchors]


def test_planes_below_the_threshold_are_not_windowed(viewer, monkeypatch, calls):
    scene = _scene(viewer, _blobs(SHAPE))
    _deselect(monkeypatch)
    viewer.dims.set_point(0, 1)
    _settle(scene)
    assert calls[-1].kwargs == {} and VispyMinimalContourOverlay._features_shape == SHAPE[1:]


@pytest.mark.parametrize("options, rgb", MODES[:3] + MODES[4:5])
def test_anchor_window_gives_the_whole_plane_contour(viewer, monkeypatch, calls, options, rgb):
    scene = _scene(viewer, _blobs(SHAPE, rgb=rgb), **options)
    _deselect(monkeypatch)
    viewer.dims.set_point(0, 1)
    _settle(scene)
    expected = _estimate(ANCHORS)
    whole = calls[-1].arrays

    monkeypatch.setattr(mc_roi, "LARGE_PLANE_PIXELS", 1000)
    monkeypatch.setattr(mc_roi, "WINDOW_SIZE", 120)
    _set_anchors(scene, ANCHORS)
    _settle(scene)
    (oy, ox), (h, w) = VispyMinimalContourOverlay._roi_origin, VispyMinimalContourOverlay._features_shape
    assert VispyMinimalContourOverlay._plane_shape == SHAPE[1:]
    assert h < SHAPE[1] and w < SHAPE[2]
    assert oy <= 101 and oy + h >= 150 and ox <= 130 and ox + w >= 190  # around the anchors
    for part, full in zip(calls[-1].arrays, whole):
        if part is not None:
            np.testing.assert_array_equal(part, full[oy:oy + h, ox:ox + w])
    _assert_same_contours(_estimate(ANCHORS), expected)


def test_anchor_window_without_anchors_is_somewhere_valid(viewer, monkeypatch, calls, small_windows):
    scene = _scene(viewer, _blobs(SHAPE))
    _deselect(monkeypatch)
    viewer.dims.set_point(0, 1)
    _settle(scene)
    (oy, ox), (h, w) = VispyMinimalContourOverlay._roi_origin, VispyMinimalContourOverlay._features_shape
    assert VispyMinimalContourOverlay._plane_shape == SHAPE[1:]
    assert h >= 80 and w >= 80 and h < SHAPE[1] and w < SHAPE[2]
    assert oy >= 0 and ox >= 0 and oy + h <= SHAPE[1] and ox + w <= SHAPE[2]


def test_anchor_window_follows_the_anchors(viewer, monkeypatch, calls, small_windows):
    scene = _scene(viewer, _blobs(SHAPE))
    _deselect(monkeypatch)
    viewer.dims.set_point(0, 1)
    near = [[40., 280.], [60., 310.], [30., 320.]]  # around the second disc
    _set_anchors(scene, near)
    _settle(scene)
    origin_near = VispyMinimalContourOverlay._roi_origin
    n_calls = len(calls)
    assert _estimate(near) is not None and len(calls) == n_calls  # inside the window: nothing is re-set

    # the cursor and the anchors move to the first disc: the window is re-centred (once)
    _estimate_far = _estimate(ANCHORS)
    assert len(calls) == n_calls + 1
    origin_far = VispyMinimalContourOverlay._roi_origin
    assert origin_far != origin_near
    (oy, ox), (h, w) = origin_far, VispyMinimalContourOverlay._features_shape
    assert oy <= 101 and oy + h >= 150 and ox <= 130 and ox + w >= 190
    assert _estimate_far is not None and sum(len(s) for s in _estimate_far) > 50
    _estimate(ANCHORS)
    assert len(calls) == n_calls + 1


def test_anchor_window_requested_contour_equals_the_whole_plane_one(viewer, monkeypatch, calls):
    scene = _scene(viewer, _blobs(SHAPE))
    _deselect(monkeypatch)
    viewer.dims.set_point(0, 1)
    _settle(scene)
    expected = _estimate(ANCHORS)

    monkeypatch.setattr(mc_roi, "LARGE_PLANE_PIXELS", 1000)
    monkeypatch.setattr(mc_roi, "WINDOW_SIZE", 80)
    _set_anchors(scene, [[40., 280.], [60., 310.], [30., 320.]])
    _settle(scene)
    _assert_same_contours(_estimate(ANCHORS), expected)


def test_object_roi_wins_over_the_anchor_window(viewer, monkeypatch, calls, small_windows):
    scene = _scene(viewer, _blobs(SHAPE))
    _select(monkeypatch, scene)
    viewer.dims.set_point(0, 1)
    _set_anchors(scene, [[40., 280.], [60., 310.]])
    _settle(scene)
    assert VispyMinimalContourOverlay._roi_origin == _expected_box().origin


def test_window_is_dropped_when_the_tool_is_turned_off(viewer, monkeypatch, calls, small_windows):
    scene = _scene(viewer, _blobs(SHAPE))
    _deselect(monkeypatch)
    viewer.dims.set_point(0, 1)
    _settle(scene)
    assert VispyMinimalContourOverlay._plane_shape == SHAPE[1:]
    scene.labels._overlays["minimal_contour"].enabled = False
    _process_events()
    assert VispyMinimalContourOverlay._features_shape is None
    assert VispyMinimalContourOverlay._roi_origin == (0, 0) and VispyMinimalContourOverlay._plane_shape is None
