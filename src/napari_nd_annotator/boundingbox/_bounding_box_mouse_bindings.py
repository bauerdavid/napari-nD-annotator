# A copy of napari.layers.shapes._shapes
from copy import copy

import numpy as np

from ._bounding_box_constants import Box, Mode


def highlight(layer, event):
    """Highlight hovered bounding boxes."""
    layer._set_highlight()


def select(layer, event):
    """Select bounding boxes or vertices either in select or direct select mode.

    Once selected bounding boxes can be moved or resized, and vertices can be moved
    depending on the mode. Holding shift when resizing a bounding box will preserve
    the aspect ratio.
    """
    shift = 'Shift' in event.modifiers
    # on press
    value = layer.get_value(event.position, world=True)
    layer._moving_value = copy(value)
    bounding_box_under_cursor, vertex_under_cursor = value
    if vertex_under_cursor is None:
        if shift and bounding_box_under_cursor is not None:
            if bounding_box_under_cursor in layer.selected_data:
                layer.selected_data.remove(bounding_box_under_cursor)
            else:
                layer.selected_data.add(bounding_box_under_cursor)
        elif bounding_box_under_cursor is not None:
            if bounding_box_under_cursor not in layer.selected_data:
                layer.selected_data = {bounding_box_under_cursor}
        else:
            layer.selected_data = set()
    layer._set_highlight()

    # we don't update the thumbnail unless a bounding box has been moved
    update_thumbnail = False
    yield

    # on move
    while event.type == 'mouse_move':
        coordinates = layer.world_to_data(event.position)
        # ToDo: Need to pass moving_coordinates to allow fixed aspect ratio
        # keybinding to work, this should be dropped
        layer._moving_coordinates = coordinates
        # Drag any selected bounding boxes
        if len(layer.selected_data) == 0:
            _drag_selection_box(layer, coordinates)
        else:
            _move(layer, coordinates)

        # if a bounding box is being moved, update the thumbnail
        if layer._is_moving:
            update_thumbnail = True
        yield

    # on release
    shift = 'Shift' in event.modifiers
    if not layer._is_moving and not layer._is_selecting and not shift:
        if bounding_box_under_cursor is not None:
            layer.selected_data = {bounding_box_under_cursor}
        else:
            layer.selected_data = set()
    elif layer._is_selecting:
        layer.selected_data = layer._data_view.bounding_boxes_in_box(layer._drag_box)
        layer._is_selecting = False
        layer._set_highlight()

    layer._is_moving = False
    layer._drag_start = None
    layer._drag_box = None
    layer._fixed_vertex = None
    layer._moving_value = (None, None)
    layer._set_highlight()

    if update_thumbnail:
        layer._update_thumbnail()


def add_bounding_box(layer, event):
    """Add a rectangle."""
    size = layer._vertex_size * layer.scale_factor / 4
    coordinates = layer.world_to_data(event.position)
    corner = np.array(coordinates).reshape(1, -1)

    data = np.tile(corner, (2**layer.ndim, 1))
    sizes = np.eye(layer.ndim)*size
    s_idx = [[]]
    for d in range(layer.ndim):
        s_idx.extend(list(e.copy()+[d] for e in s_idx))
    addition = np.asarray([sizes[idx].sum(0) for idx in s_idx])
    data = data + addition
    data = data[np.newaxis]
    yield from _add_bounding_box(
        layer, event, data=data
    )


def _add_bounding_box(layer, event, data):
    """Helper function for adding a bounding box."""
    # on press
    # Start drawing rectangle / ellipse / line
    layer.add(data)
    layer.selected_data = {layer.nbounding_boxes - 1}
    layer._value = (layer.nbounding_boxes - 1, 4)
    layer._moving_value = copy(layer._value)
    layer.refresh()
    yield

    data = layer.data[layer.nbounding_boxes - 1]
    # on move
    const_set = False
    while event.type == 'mouse_move':
        # Drag any selected bounding boxes
        coordinates = layer.world_to_data(event.position)
        _move(layer, coordinates)
        min = data.min(0)
        max = data.max(0)
        size = max-min
        visible_size = size[layer._dims_displayed]
        max[layer._dims_displayed] = np.nan
        min[layer._dims_displayed] = np.nan
        if layer.size_mode == "average":
            data[:] = np.where(data == max, coordinates+visible_size.mean()/2*layer.size_multiplier, data)
            data[:] = np.where(data == min, coordinates-visible_size.mean()/2*layer.size_multiplier, data)
        elif not const_set and layer.size_mode == "constant":
            data[:] = np.where(data == max, np.asarray(coordinates) + layer.size_constant/2, data)
            data[:] = np.where(data == min, np.asarray(coordinates) - layer.size_constant/2, data)
            const_set = True
        yield

    # on release
    layer._finish_drawing()


def _drag_selection_box(layer, coordinates):
    """Drag a selection box.

    Parameters
    ----------
    layer : .bounding_boxes.BoundingBoxLayer
        Bounding Box layer.
    coordinates : tuple
        Position of mouse cursor in data coordinates.
    """
    # If something selected return
    if len(layer.selected_data) > 0:
        return

    coord = [coordinates[i] for i in layer._dims_displayed]

    # Create or extend a selection box
    layer._is_selecting = True
    if layer._drag_start is None:
        layer._drag_start = coord
    layer._drag_box = np.array([layer._drag_start, coord])
    layer._set_highlight()


def _move(layer, coordinates):
    """Moves object at given mouse position and set of indices.

    Parameters
    ----------
    layer : BoundingBoxLayer
        BoundingBoxLayer layer.
    coordinates : tuple
        Position of mouse cursor in data coordinates.
    """
    # If nothing selected return
    if len(layer.selected_data) == 0:
        return

    vertex = layer._moving_value[1]

    if layer._mode in (
        [Mode.SELECT, Mode.ADD_BOUNDING_BOX]
    ):
        coord = [coordinates[i] for i in layer._dims_displayed]
        layer._moving_coordinates = coordinates
        layer._is_moving = True
        if vertex is None:
            # Check where dragging box from to move whole object
            if layer._drag_start is None:
                center = layer._selected_box[Box.CENTER]
                layer._drag_start = coord - center
            center = layer._selected_box[Box.CENTER]
            shift = coord - center - layer._drag_start
            for index in layer.selected_data:
                layer._data_view.shift(index, shift)
            layer._selected_box = layer._selected_box + shift
            layer.refresh()
        elif vertex < Box.LEN:
            # Corner / edge vertex is being dragged so resize object
            box = layer._selected_box
            if layer._fixed_vertex is None:
                layer._fixed_index = (vertex + 4) % Box.LEN
                layer._fixed_vertex = box[layer._fixed_index]

            size = (
                box[(layer._fixed_index + 4) % Box.LEN]
                - box[layer._fixed_index]
            )
            offset = box[Box.HANDLE] - box[Box.CENTER]
            if np.linalg.norm(offset) == 0:
                offset = [1, 1]
            offset = offset / np.linalg.norm(offset)
            offset_perp = np.array([offset[1], -offset[0]])

            fixed = layer._fixed_vertex
            new = list(coord)

            if layer._fixed_aspect and layer._fixed_index % 2 == 0:
                if (new - fixed)[0] == 0:
                    ratio = 1
                else:
                    ratio = abs((new - fixed)[1] / (new - fixed)[0])
                if ratio > layer._aspect_ratio:
                    r = layer._aspect_ratio / ratio
                    new[1] = fixed[1] + (new[1] - fixed[1]) * r
                else:
                    r = ratio / layer._aspect_ratio
                    new[0] = fixed[0] + (new[0] - fixed[0]) * r

            # if size @ offset == 0:
            if np.allclose(size @ offset, 0):
                dist = 1
            else:
                dist = ((new - fixed) @ offset) / (size @ offset)

            if size @ offset_perp == 0:
                dist_perp = 1
            else:
                dist_perp = ((new - fixed) @ offset_perp) / (
                    size @ offset_perp
                )

            if layer._fixed_index % 2 == 0:
                # corner selected
                scale = np.array([dist_perp, dist])
            elif layer._fixed_index % 4 == 3:
                # top selected
                scale = np.array([1, dist])
            else:
                # side selected
                scale = np.array([dist_perp, 1])

            # prevent box from shrinking below a threshold size
            threshold = layer._vertex_size * layer.scale_factor / 8
            scale[abs(scale * size[[1, 0]]) < threshold] = 1

            # check orientation of box
            angle = -np.arctan2(offset[0], -offset[1])
            c, s = np.cos(angle), np.sin(angle)
            # if angle == 0:
            if np.allclose(angle, 0):
                for index in layer.selected_data:
                    layer._data_view.scale(
                        index, scale, center=layer._fixed_vertex
                    )
                layer._scale_box(scale, center=layer._fixed_vertex)
            else:
                rotation = np.array([[c, s], [-s, c]])
                scale_mat = np.array([[scale[0], 0], [0, scale[1]]])
                inv_rot = np.array([[c, -s], [s, c]])
                transform = rotation @ scale_mat @ inv_rot
                for index in layer.selected_data:
                    layer._data_view.shift(index, -layer._fixed_vertex)
                    layer._data_view.transform(index, transform)
                    layer._data_view.shift(index, layer._fixed_vertex)
                layer._transform_box(transform, center=layer._fixed_vertex)
            layer.refresh()
        elif vertex == 8:
            # Rotation handle is being dragged so rotate object
            handle = layer._selected_box[Box.HANDLE]
            if layer._drag_start is None:
                layer._fixed_vertex = layer._selected_box[Box.CENTER]
                offset = handle - layer._fixed_vertex
                layer._drag_start = -np.degrees(
                    np.arctan2(offset[0], -offset[1])
                )

            new_offset = coord - layer._fixed_vertex
            new_angle = -np.degrees(np.arctan2(new_offset[0], -new_offset[1]))
            fixed_offset = handle - layer._fixed_vertex
            fixed_angle = -np.degrees(
                np.arctan2(fixed_offset[0], -fixed_offset[1])
            )

            if np.linalg.norm(new_offset) < 1:
                angle = 0
            elif layer._fixed_aspect:
                angle = np.round(new_angle / 45) * 45 - fixed_angle
            else:
                angle = new_angle - fixed_angle

            for index in layer.selected_data:
                layer._data_view.rotate(
                    index, angle, center=layer._fixed_vertex
                )
            # layer._rotate_box(angle, center=layer._fixed_vertex)
            layer.refresh()
    elif layer._mode == Mode.DIRECT:
        if vertex is not None:
            layer._is_moving = True
            index = layer._moving_value[0]
            vertices = layer._data_view.bounding_boxes[index].data
            vertices[vertex] = coordinates
            layer._data_view.edit(index, vertices)
            bounding_boxes = layer.selected_data
            layer._selected_box = layer.interaction_box(bounding_boxes)
            layer.refresh()
