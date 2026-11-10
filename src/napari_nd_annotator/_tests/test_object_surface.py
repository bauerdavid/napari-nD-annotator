"""The Minimal Surface widget working per object: anchors of the selected object, placing them in the 3D view and
writing the result with the label id of the object.

Needs a GL-capable Qt platform, the optional minimal_surface package and the napari-bbox plugin (like the other viewer
tests, skipped when one is missing). The running of an estimation is in ``test_minimal_surface_widget.py``.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("minimal_surface")
pytest.importorskip("napari_bbox")

from .test_napari_compat import viewer, _process_events  # noqa: E402,F401
from .test_suggest_anchors_widget import (  # noqa: E402,F401
    annotator, scene, _select, _anchors, _distance_from_ball_surface, BALL_CENTER, BALL_RADIUS, SHAPE)

from napari_nd_annotator._widgets.minimal_surface_widget import write_object_mask  # noqa: E402
from napari_nd_annotator._widgets.object_list import selected_object_info  # noqa: E402
from napari_nd_annotator.object_anchors import anchor_store  # noqa: E402

LABEL = 3  # the label of the object of the scene


def _click(ms, viewer, position, direction, button=1, drag=False, modifiers=()):
    """A click of the mouse in the 3D canvas, through the callback that napari calls."""
    event = SimpleNamespace(type="mouse_press", button=button, modifiers=modifiers, position=tuple(position),
                            view_direction=tuple(np.asarray(direction, dtype=float)), dims_displayed=(0, 1, 2))
    callback = ms._on_canvas_mouse(viewer, event)
    next(callback, None)
    if drag:
        event.type = "mouse_move"
        next(callback, None)
    event.type = "mouse_release"
    for _ in callback:
        pass


@pytest.fixture
def surface_tab(annotator):
    """The Minimal Surface tab is the shown one (anchors are placed only then)."""
    annotator.ToolsWidget.current_index = 2
    yield annotator.ToolsWidget.minimal_surface_widget


@pytest.fixture
def object_scene(viewer, scene, surface_tab):
    """The scene with the object selected, in the 3D view."""
    image_layer, object_list, ms = scene
    viewer.dims.ndisplay = 3
    _select(viewer, object_list)
    return image_layer, object_list, ms


def test_clicks_place_the_anchors_on_the_surface(viewer, object_scene):
    image_layer, object_list, ms = object_scene
    assert len(_anchors(ms)) == 0
    # from both sides along x, through the center of the ball
    _click(ms, viewer, (20, 30, -20), (0, 0, 1))
    assert len(_anchors(ms)) == 1
    _click(ms, viewer, (20, 30, 80), (0, 0, -1))
    points = _anchors(ms)
    assert points.shape == (2, 3)
    assert np.all(_distance_from_ball_surface(points) < 1.5)
    assert points[0, 2] < BALL_CENTER[2] < points[1, 2]
    assert ms.object_status_label.value == "%d (2/2 anchors)" % LABEL
    # the plugin's display layer shows them, in world coordinates
    np.testing.assert_allclose(ms._anchors_layer.data, [image_layer.data_to_world(p) for p in points])
    assert not ms._anchors_layer.editable
    # a third click starts a new pair
    _click(ms, viewer, (20, 30, -20), (0, 0, 1))
    assert len(_anchors(ms)) == 1


def test_right_click_removes_the_last_anchor(viewer, object_scene):
    image_layer, object_list, ms = object_scene
    _click(ms, viewer, (20, 30, -20), (0, 0, 1))
    _click(ms, viewer, (20, 30, 80), (0, 0, -1))
    _click(ms, viewer, (0, 0, 0), (0, 0, 1), button=2)
    assert len(_anchors(ms)) == 1
    ms.clear_anchors_button.native.click()
    assert len(_anchors(ms)) == 0
    assert not ms._anchors_layer.visible


def test_ray_that_misses_the_object_places_nothing(viewer, object_scene):
    image_layer, object_list, ms = object_scene
    _click(ms, viewer, (20, 5, -20), (0, 0, 1))  # beside the ball
    _click(ms, viewer, (39, 59, -20), (0, 0, 1))  # outside the bounding box
    assert len(_anchors(ms)) == 0


def test_anchors_are_picked_in_the_clipped_box_only(viewer, object_scene):
    image_layer, object_list, ms = object_scene
    # a bright wall in front of the box is not hit
    image_layer.data[:, :, :8] = image_layer.data.max()
    _click(ms, viewer, (20, 30, -20), (0, 0, 1))
    assert abs(_anchors(ms)[0, 2] - (BALL_CENTER[2] - BALL_RADIUS)) < 1.5


@pytest.mark.parametrize("case", ["drag", "modifier", "checkbox off", "2D", "other tab", "no object", "middle button"])
def test_clicks_that_are_not_for_the_anchors_are_ignored(viewer, object_scene, annotator, case):
    image_layer, object_list, ms = object_scene
    kwargs = {}
    if case == "drag":
        kwargs["drag"] = True
    elif case == "modifier":
        kwargs["modifiers"] = ("Shift",)
    elif case == "checkbox off":
        ms.place_anchors_checkbox.value = False
    elif case == "2D":
        viewer.dims.ndisplay = 2
    elif case == "other tab":
        annotator.ToolsWidget.current_index = 0
    elif case == "no object":
        object_list.select_item(object_list.item(0))  # the second click unselects it
        _process_events()
        assert selected_object_info(viewer) is None
    elif case == "middle button":
        kwargs["button"] = 3
    _click(ms, viewer, (20, 30, -20), (0, 0, 1), **kwargs)
    assert len(anchor_store(object_list.bounding_box_layer).get(LABEL)) == 0


def test_image_plane_placement(viewer, object_scene):
    image_layer, object_list, ms = object_scene
    ms.anchor_placement_combobox.value = "Image plane"
    assert image_layer.depiction == "plane"
    # the plane is moved into the bounding box, with its normal along the first axis
    position = np.asarray(image_layer.plane.position)
    assert np.all(position >= object_list.item(0).bounding_box[0]) and np.all(position <= object_list.item(0).bounding_box[1])
    image_layer.plane.position = (22., 30., 30.)
    image_layer.plane.normal = (1, 0, 0)
    # a ray along the first axis crosses the plane at its own y, x
    _click(ms, viewer, (-20, 26, 28), (1, 0, 0))
    np.testing.assert_allclose(_anchors(ms), [[22, 26, 28]], atol=1e-6)
    # a ray that crosses the plane outside the bounding box places nothing
    image_layer.plane.position = (0., 30., 30.)
    _click(ms, viewer, (-20, 26, 28), (1, 0, 0))
    assert len(_anchors(ms)) == 1
    ms.anchor_placement_combobox.value = "Ray pick"
    assert image_layer.depiction == "volume"
    _click(ms, viewer, (20, 30, 80), (0, 0, -1))
    assert abs(_anchors(ms)[1, 2] - (BALL_CENTER[2] + BALL_RADIUS)) < 1.5


def _add_second_object(object_list):
    import itertools
    layer = object_list.bounding_box_layer
    corners = np.where(list(itertools.product((False, True), repeat=3)), [30, 45, 45], [10, 15, 15]).astype(float)
    layer.data = [np.asarray(box) for box in layer.data] + [corners]
    layer.features = {"label": np.array([LABEL, LABEL + 1])}
    object_list.update_items()
    _process_events()


def test_anchors_belong_to_their_object(viewer, scene, surface_tab):
    image_layer, object_list, ms = scene
    viewer.dims.ndisplay = 3
    _select(viewer, object_list)
    _click(ms, viewer, (20, 30, -20), (0, 0, 1))
    _click(ms, viewer, (20, 30, 80), (0, 0, -1))
    first = _anchors(ms).copy()
    _add_second_object(object_list)
    object_list.select_item(object_list.item(1))
    _process_events()
    selected = selected_object_info(viewer)
    assert selected.label == LABEL + 1 and object_list.count() == 2
    assert len(_anchors(ms)) == 0 and ms.object_status_label.value.startswith("%d (0/2" % (LABEL + 1))
    assert not ms._anchors_layer.visible
    # anchors of the new object do not touch those of the first
    _click(ms, viewer, (20, 30, 80), (0, 0, -1))
    assert len(_anchors(ms)) == 1
    object_list.select_item(object_list.item(0))
    _process_events()
    np.testing.assert_allclose(_anchors(ms), first)
    assert ms._anchors_layer.visible and len(ms._anchors_layer.data) == 2
    # the layers of the object stay the selected ones
    assert ms._anchors_layer not in viewer.layers.selection
    assert object_list.mask_layer in viewer.layers.selection


def test_anchors_are_stored_with_the_bounding_box_layer(viewer, object_scene):
    image_layer, object_list, ms = object_scene
    _click(ms, viewer, (20, 30, -20), (0, 0, 1))
    stored = object_list.bounding_box_layer.metadata["nd_annotator_anchors"]
    assert list(stored) == [str(LABEL)] and len(stored[str(LABEL)]) == 1


def test_run_needs_an_object_with_two_anchors(viewer, scene, surface_tab):
    image_layer, object_list, ms = scene
    with pytest.warns(UserWarning, match="No object selected"):
        ms._start_estimation()
    viewer.dims.ndisplay = 3
    _select(viewer, object_list)
    with pytest.warns(UserWarning, match="needs two anchors"):
        ms._start_estimation()
    _click(ms, viewer, (20, 30, -20), (0, 0, 1))
    with pytest.warns(UserWarning, match="needs two anchors"):
        ms._start_estimation()
    assert ms.call_button.text == "Run"


def test_run_checks_the_labels_layer_of_the_object_list(viewer, scene, surface_tab):
    image_layer, object_list, ms = scene
    viewer.dims.ndisplay = 3
    _select(viewer, object_list)
    _click(ms, viewer, (20, 30, -20), (0, 0, 1))
    _click(ms, viewer, (20, 30, 80), (0, 0, -1))
    object_list.mask_layer.data = np.zeros((10, 10, 10), dtype=np.uint16)
    with pytest.warns(UserWarning, match="different shapes"):
        ms._start_estimation()
    assert ms.call_button.text == "Run"
    object_list.mask_layer.data = np.zeros(SHAPE, dtype=np.uint16)


def _labels(viewer, dtype=np.uint16):
    return viewer.add_labels(np.zeros(SHAPE, dtype=dtype), name="written")


def _ball_mask():
    zz, yy, xx = np.mgrid[:20, :20, :20]
    return ((zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) <= 64


def test_object_mask_is_written_with_the_label_id_and_can_be_undone(viewer):
    labels = _labels(viewer)
    mask = _ball_mask()
    write_object_mask(labels, 7, mask, np.array([5., 8., 9.]))
    assert np.array_equal(labels.data == 7, np.pad(mask, [(5, 15), (8, 32), (9, 31)]))
    assert set(np.unique(labels.data)) == {0, 7}
    labels.undo()
    assert labels.data.max() == 0
    labels.redo()
    assert np.count_nonzero(labels.data == 7) == np.count_nonzero(mask)


def test_object_mask_replaces_only_the_previous_mask_of_the_object(viewer):
    labels = _labels(viewer)
    labels.data[0:5, 0:5, 0:5] = 2  # another object
    labels.data[30:35, 30:35, 30:35] = 7  # the previous mask of the object, inside its box
    labels.data[20:22, 50:52, 50:52] = 7  # a stray part far away from the box and the new mask
    labels.data[14:16, 17:19, 18:20] = 5  # another object under the new mask
    box = (np.array([25, 25, 25]), np.array([38, 38, 38]))
    write_object_mask(labels, 7, _ball_mask(), np.array([5., 8., 9.]), box)
    data = np.asarray(labels.data)
    assert not np.any(data[30:35, 30:35, 30:35] == 7)  # replaced
    assert np.all(data[0:5, 0:5, 0:5] == 2)  # others stay
    assert np.all(data[20:22, 50:52, 50:52] == 7)  # outside the touched region
    assert np.all(data[14:16, 17:19, 18:20] == 7)  # painted over
    assert data[15, 18, 19] == 7  # the new mask
    # one undo restores everything
    labels.undo()
    assert np.all(labels.data[30:35, 30:35, 30:35] == 7) and np.all(labels.data[14:16, 17:19, 18:20] == 5)
    assert labels.data[15, 18, 19] == 5


def test_object_mask_is_cut_at_the_edge_of_the_layer(viewer):
    labels = _labels(viewer)
    write_object_mask(labels, 4, _ball_mask(), np.array([30., 50., 50.]))
    assert np.count_nonzero(labels.data == 4) == np.count_nonzero(_ball_mask()[:10, :10, :10])
    # an empty result changes nothing and adds no history
    before = len(labels._undo_history)
    write_object_mask(labels, 4, np.zeros((3, 3, 3), dtype=bool), np.array([0., 0., 0.]), None)
    labels.undo()
    labels.undo()
    assert len(labels._undo_history) == 0 and before == 1 and labels.data.max() == 0


def test_settings_saved_with_the_old_widget_still_load(viewer, annotator, tmp_path):
    import json
    from magicclass.serialize import serialize
    ms = annotator.ToolsWidget.minimal_surface_widget
    assert not hasattr(ms, "use_correction") and not hasattr(ms, "labels_layer_combobox")
    data = serialize(annotator)
    saved = data["ToolsWidget"]["Minimal Surface"]
    # the settings are stored both flat and by container
    for container in (saved, saved["Image Features"]):
        container.update({"use_correction": False, "alpha": 0.2, "z_scale": 3.})
    saved["labels_layer_combobox"] = 1
    path = tmp_path / "settings"
    path.write_text(json.dumps(data))
    annotator._load(path)
    assert ms.alpha == pytest.approx(.2) and ms.z_scale == pytest.approx(3.)





def test_writing_the_mask_in_3D_does_not_upset_the_interpolation_widget(viewer, object_scene):
    # data_setitem emits labels_update like painting does; the labels layer is shown in 3D, so it has no slice dimension
    image_layer, object_list, ms = object_scene
    labels = object_list.mask_layer
    assert viewer.layers.selection.active is labels and viewer.dims.ndisplay == 3
    write_object_mask(labels, LABEL, _ball_mask(), np.array([5., 8., 9.]))
    _process_events()
    assert np.count_nonzero(labels.data == LABEL) == np.count_nonzero(_ball_mask())
