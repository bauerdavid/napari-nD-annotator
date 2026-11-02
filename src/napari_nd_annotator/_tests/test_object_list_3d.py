"""In the 3D view the original layers are clipped to the selected object (no cropped copies are created).

Like the other viewer tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools

import numpy as np
import pytest

pytest.importorskip("napari_bbox")

from napari_nd_annotator._helper_functions import viewer_camera  # noqa: E402
from napari_nd_annotator._widgets._utils.clipping import SCAFFOLD_KEY, is_scaffold  # noqa: E402
from .test_object_list import (  # noqa: E402,F401  (the fixtures are used by their names)
    BBOX_MAX, BBOX_MIN, OBJECT_ID, SCALE, TRANSLATE,
    _add_objects, _delete_box, _process_events, _select_object, scene, viewer,
)


def _kept_voxels(viewer, layer):
    """Boolean mask of the voxels (centers) of the layer that its clipping planes keep.

    The planes are in world coordinates, in the order of the displayed axes.
    """
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in layer.data.shape], indexing="ij"), axis=-1).astype(float)
    world = grid * layer.scale + layer.translate
    kept = np.ones(layer.data.shape, dtype=bool)
    displayed = list(viewer.dims.displayed)
    for plane in layer.experimental_clipping_planes:
        offset = world[..., displayed] - np.asarray(plane.position)
        kept &= (offset @ np.asarray(plane.normal)) >= 0
    return kept


def _bbox_voxels(shape, box_min=BBOX_MIN, box_max=BBOX_MAX):
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(lo, hi + 1) for lo, hi in zip(box_min, box_max))] = True
    return mask


def _scaffold_layers(viewer):
    return [layer for layer in viewer.layers if is_scaffold(layer)]


def _select_3d(viewer, labels, object_list):
    viewer.dims.ndisplay = 3
    _process_events()
    _select_object(viewer, labels, object_list)


def test_selection_clips_the_original_layers_to_the_bounding_box(viewer, scene):
    image, labels, object_list = scene
    _select_3d(viewer, labels, object_list)
    assert not [layer for layer in viewer.layers if layer.name.startswith("Cropped")]
    assert object_list.projections_widget is None
    for layer in (image, labels):
        assert len(layer.experimental_clipping_planes) == 6
        np.testing.assert_array_equal(_kept_voxels(viewer, layer), _bbox_voxels(layer.data.shape))
        center = np.asarray(layer.data_to_world((BBOX_MIN + BBOX_MAX) / 2))[list(viewer.dims.displayed)]
        for plane in layer.experimental_clipping_planes:  # the normal points into the kept region
            assert np.count_nonzero(plane.normal) == 1
            assert np.dot(center - np.asarray(plane.position), plane.normal) > 0
        # the first plane is the lower one of the first axis, half a voxel outside the box, in world coordinates
        first = layer.experimental_clipping_planes[0]
        np.testing.assert_allclose(first.position[0], (BBOX_MIN[0] - .5) * SCALE[0] + TRANSLATE[0])


def test_selection_with_a_transposed_view(viewer, scene):
    image, labels, object_list = scene
    viewer.dims.order = (1, 0, 2)
    _select_3d(viewer, labels, object_list)
    for layer in (image, labels):
        np.testing.assert_array_equal(_kept_voxels(viewer, layer), _bbox_voxels(layer.data.shape))


def test_selection_selects_the_original_labels_and_moves_the_view(viewer, scene):
    image, labels, object_list = scene
    _select_3d(viewer, labels, object_list)
    assert list(viewer.layers.selection) == [labels]
    assert labels.selected_label == OBJECT_ID
    center = image.data_to_world((BBOX_MIN + BBOX_MAX) / 2)
    np.testing.assert_allclose(viewer_camera(viewer).center, center)
    np.testing.assert_allclose(viewer.dims.point, center, atol=max(SCALE) / 2)  # the sliders snap to their steps


def test_unselect_restores_clipping_planes_and_visibility(viewer, scene):
    image, labels, object_list = scene
    image.experimental_clipping_planes = [{"position": (0, 0, 7), "normal": (0, 0, 1)}]
    image.visible = False
    _select_3d(viewer, labels, object_list)
    assert image.visible and len(image.experimental_clipping_planes) == 6
    object_list.select_item(object_list.item(0))  # clicking the selected object again unselects it
    _process_events()
    assert object_list.selected_idx is None
    assert [(tuple(p.position), tuple(p.normal)) for p in image.experimental_clipping_planes] \
        == [((0, 0, 7), (0, 0, 1))]
    assert not image.visible and labels.visible
    assert len(labels.experimental_clipping_planes) == 0
    assert not _scaffold_layers(viewer)


def test_selecting_another_object_moves_the_clipping(viewer, scene):
    image, labels, object_list = scene
    _add_objects(object_list, [5])
    _select_3d(viewer, labels, object_list)
    object_list.select_item(object_list.item(1))
    _process_events()
    second = object_list.item(1).bounding_box
    np.testing.assert_array_equal(_kept_voxels(viewer, labels), _bbox_voxels(labels.data.shape, *second))
    assert labels.selected_label == 5
    assert len(_scaffold_layers(viewer)) == 1  # the ghost of the first object was replaced, not duplicated


def test_clipping_follows_the_bounding_box(viewer, scene):
    image, labels, object_list = scene
    _select_3d(viewer, labels, object_list)
    shift = np.array([1, 2, 3])
    corners = np.where(list(itertools.product((False, True), repeat=3)), BBOX_MAX + shift, BBOX_MIN + shift)
    object_list.bounding_box_layer.data = [(corners * SCALE + TRANSLATE).astype(float)]
    object_list.update_items()
    _process_events()
    for layer in (image, labels):
        np.testing.assert_array_equal(_kept_voxels(viewer, layer),
                                      _bbox_voxels(layer.data.shape, BBOX_MIN + shift, BBOX_MAX + shift))


def test_deleting_the_bounding_box_removes_the_clipping(viewer, scene):
    image, labels, object_list = scene
    _add_objects(object_list, [5])
    _select_3d(viewer, labels, object_list)
    _delete_box(object_list, 0)
    assert object_list.selected_idx is None and object_list.clipper is None
    for layer in (image, labels):
        assert len(layer.experimental_clipping_planes) == 0
    assert not _scaffold_layers(viewer)


def test_ghost_shares_the_data_and_is_not_editable(viewer, scene):
    image, labels, object_list = scene
    _select_3d(viewer, labels, object_list)
    ghosts = _scaffold_layers(viewer)
    assert len(ghosts) == 1  # by default only the image has a ghost
    ghost = ghosts[0]
    assert ghost.metadata[SCAFFOLD_KEY] is True
    assert type(ghost) is type(image)
    assert ghost.data is image.data
    assert len(ghost.experimental_clipping_planes) == 0
    assert ghost.opacity < image.opacity
    assert not ghost.editable
    assert ghost not in viewer.layers.selection
    assert viewer.layers.index(ghost) < viewer.layers.index(image)
    np.testing.assert_allclose(ghost.scale, image.scale)
    np.testing.assert_allclose(ghost.translate, image.translate)


def test_ghost_follows_the_original(viewer, scene):
    image, labels, object_list = scene
    object_list.show_labels_surroundings = True
    _select_3d(viewer, labels, object_list)
    image_ghost, labels_ghost = [next(g for g in _scaffold_layers(viewer) if type(g) is type(layer))
                                 for layer in (image, labels)]
    assert labels_ghost.data is labels.data
    image.contrast_limits = (0.2, 0.7)
    image.colormap = "viridis"
    ratio = image_ghost.opacity / image.opacity
    image.opacity = 0.8
    assert tuple(image_ghost.contrast_limits) == (0.2, 0.7)
    assert image_ghost.colormap.name == "viridis"
    assert image_ghost.opacity == pytest.approx(0.8 * ratio)
    new_data = np.zeros_like(image.data)
    image.data = new_data
    assert image_ghost.data is new_data
    assert len(image_ghost.experimental_clipping_planes) == 0
    image.visible = False
    assert not image_ghost.visible


def test_ghosts_can_be_toggled_while_selected(viewer, scene):
    image, labels, object_list = scene
    _select_3d(viewer, labels, object_list)
    assert len(_scaffold_layers(viewer)) == 1
    object_list.show_labels_surroundings = True
    assert len(_scaffold_layers(viewer)) == 2
    object_list.show_image_surroundings = False
    assert [type(g) for g in _scaffold_layers(viewer)] == [type(labels)]
    assert list(viewer.layers.selection) == [labels]
    object_list.show_labels_surroundings = False
    assert not _scaffold_layers(viewer)


def test_ghosts_are_removed_when_the_original_is_removed(viewer, scene):
    image, labels, object_list = scene
    object_list.show_labels_surroundings = True
    _select_3d(viewer, labels, object_list)
    assert len(_scaffold_layers(viewer)) == 2
    viewer.layers.remove(image)
    assert [type(g) for g in _scaffold_layers(viewer)] == [type(labels)]
    viewer.layers.remove(labels)
    assert not _scaffold_layers(viewer)


def test_closing_the_viewer_with_a_selected_object(viewer, scene):
    image, labels, object_list = scene
    object_list.show_labels_surroundings = True
    _select_3d(viewer, labels, object_list)
    viewer.close()
    _process_events()
    assert not _scaffold_layers(viewer)


def test_swapping_between_2d_and_3d_while_an_object_is_selected(viewer, scene):
    image, labels, object_list = scene
    _select_3d(viewer, labels, object_list)
    viewer.dims.ndisplay = 2
    _process_events()
    # back to the plane view of the original layers, without clipping and ghosts
    assert object_list.projections_widget is not None
    assert object_list.clipper is None
    assert not _scaffold_layers(viewer)
    for layer in (image, labels):
        assert len(layer.experimental_clipping_planes) == 0
        assert layer.visible
    assert object_list.selected_idx == 0
    viewer.dims.ndisplay = 3
    _process_events()
    assert not [layer for layer in viewer.layers if layer.name.startswith("Cropped")]
    assert object_list.projections_widget is None
    for layer in (image, labels):
        assert layer.visible
        np.testing.assert_array_equal(_kept_voxels(viewer, layer), _bbox_voxels(layer.data.shape))
    assert len(_scaffold_layers(viewer)) == 1


def test_2d_selection_neither_clips_nor_creates_layers(viewer, scene):
    image, labels, object_list = scene
    assert viewer.dims.ndisplay == 2
    layers = list(viewer.layers)
    _select_object(viewer, labels, object_list)
    assert list(viewer.layers) == layers
    assert object_list.clipper is None and not _scaffold_layers(viewer)
    assert image.visible and labels.visible
    for layer in (image, labels):
        assert len(layer.experimental_clipping_planes) == 0
    assert list(viewer.layers.selection) == [labels]


def test_the_clipped_region_is_what_the_3d_canvas_shows(viewer, scene):
    """The planes act in the canvas: a quarter of the labels is left, not the other three quarters."""
    image, labels, object_list = scene
    image.opacity = 0  # only the labels are seen
    labels.data[:] = OBJECT_ID
    labels.rendering = "translucent"  # the shading of the default isosurface rendering darkens parts of the clipped region
    labels.refresh()
    object_list.show_image_surroundings = False
    object_list.bounding_box_layer.visible = False
    viewer.window.show()  # nothing is rendered into the screenshot of a hidden window
    viewer.dims.ndisplay = 3
    _process_events()
    viewer.reset_view()
    # the camera looks along the first axis, the object is a quarter of the last one
    box_min, box_max = [0, 0, 0], [9, 39, 9]
    corners = np.where(list(itertools.product((False, True), repeat=3)), box_max, box_min)
    object_list.bounding_box_layer.data = [(corners * SCALE + TRANSLATE).astype(float)]
    object_list.update_items()
    _process_events()

    def painted():
        screenshot = viewer.screenshot(canvas_only=True, flash=False)[..., :3].astype(int)
        return np.count_nonzero(screenshot.max(-1) > 0)  # the background is black

    unclipped = painted()
    assert unclipped > 0
    _select_3d(viewer, labels, object_list)
    clipped = painted()
    assert clipped / unclipped == pytest.approx(0.25, abs=0.03)  # the edges of the rendered volume are a bit darker


def test_surroundings_checkboxes_control_the_ghosts(viewer, scene):
    from napari_nd_annotator import ListWidgetBB
    image, labels, _ = scene
    widget = ListWidgetBB(viewer)
    assert widget.show_image_surroundings_checkbox.isChecked()
    assert not widget.show_labels_surroundings_checkbox.isChecked()
    widget.image.layer, widget.labels.layer = image, labels
    object_list = widget.list_widget
    assert object_list.show_image_surroundings and not object_list.show_labels_surroundings
    _select_3d(viewer, labels, object_list)
    assert [type(g) for g in _scaffold_layers(viewer)] == [type(image)]
    widget.show_labels_surroundings_checkbox.setChecked(True)
    widget.show_image_surroundings_checkbox.setChecked(False)
    assert [type(g) for g in _scaffold_layers(viewer)] == [type(labels)]
