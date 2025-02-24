import traceback
import warnings
import inspect
from functools import lru_cache

import numpy as np
import skimage
from PyQt5.QtCore import QMutex, Qt
from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari._vispy.utils import visual
from napari.layers import Labels
from napari.layers.labels._labels_constants import Mode
from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate
from napari.settings import get_settings
from napari_nd_annotator.minimal_contour._eikonal_wrapper import MinimalContourCalculator
from vispy.scene import Polygon, Line, Compound, Markers

from napari_nd_annotator._helper_functions import layer_get_order

from .minimal_contour_overlay import MinimalContourOverlay

def bbox_around_points(pts):
    p1 = pts.min(0)
    p2 = pts.max(0)
    size = p2 - p1
    from_i = p1 - size[0]*0.1 - 10
    to_i = p2 + size[0]*0.1 + 10
    return from_i, to_i


def _only_when_enabled(callback):
    """Decorator that wraps a callback of VispyLabelsPolygonOverlay.

    It ensures that the callback is only executed when all the conditions are met:
    1) The overlay is enabled;
    2) The number of displayed dimensions is 2 (it can only work in 2D);
    3) The number of dimensions across which labels will be edited is 2.

    If 2, 3 are not met, the Labels mode is automatically switched to PAN_ZOOM.
    """
    if inspect.isgeneratorfunction(callback):
        def decorated_callback(self, layer: Labels, event):
            if not self.overlay.enabled:
                return

            if self._features_shape is None:
                return

            if layer._slice_input.ndisplay != 2 or layer.n_edit_dimensions != 2:
                return
            yield from callback(self, layer, event)
    else:
        def decorated_callback(self, layer: Labels, event):
            if not self.overlay.enabled:
                return

            if self._features_shape is None:
                return

            if layer._slice_input.ndisplay != 2 or layer.n_edit_dimensions != 2:
                return
            callback(self, layer, event)

    return decorated_callback


cache_size = 20_000
dash_len = 1
skip_len = 7
dashed_cache = np.concatenate([np.arange(i, i+dash_len, dtype=np.uint32) for i in range(0, cache_size*dash_len, dash_len+skip_len)])
dashed_cache = np.tile(dashed_cache[:, None], (1, 2))
dashed_cache[:, 1] += 4


@lru_cache(maxsize=200)
def get_dashed_cache(data_len):
    return dashed_cache[:np.searchsorted(dashed_cache[:, 1], data_len)]


class VispyMinimalContourOverlay(LayerOverlayMixin, VispySceneOverlay):
    layer: Labels
    mc_calculator = MinimalContourCalculator(None, 3)
    _features_shape = None
    def __init__(
            self, *, layer: Labels, overlay: MinimalContourOverlay, parent=None
    ):
        self._features = None
        points = [(0, 0), (1, 1)]
        self.point_triangle = np.zeros((3, 2), dtype=np.float64) - 1  # start point, current position, end point

        self._nodes_kwargs = {
            'face_color': (1, 1, 1, 1),
            'size': 8.0,
            'edge_width': 1.0,
            'edge_color': (0, 0, 0, 1),
            'symbol': 'x'
        }

        self._anchor_points = Markers(pos=np.array(points), **self._nodes_kwargs)
        self._last_anchor_to_current_pos_contour = Line(pos=points, width=overlay.contour_width, color=(1., 0., 0., 1.), method='gl')
        self._current_pos_to_first_anchor_contour = Line(pos=points, width=overlay.contour_width, color=(.3, .3, .3, 1.), method='gl')
        self._stored_contour = Line(pos=points, width=overlay.contour_width, method='agg')
        super().__init__(
            node=Compound([self._anchor_points, self._last_anchor_to_current_pos_contour, self._current_pos_to_first_anchor_contour, self._stored_contour]),
            layer=layer,
            overlay=overlay,
            parent=parent,
        )

        self.layer.mouse_move_callbacks.append(self._on_mouse_move)
        self.layer.mouse_drag_callbacks.append(self._on_mouse_press)
        self.layer.mouse_double_click_callbacks.append(
            self._on_mouse_double_click
        )

        self.overlay.events.anchor_points.connect(self._on_points_change)
        self.overlay.events.enabled.connect(self._on_enabled_change)
        self.overlay.events.use_straight_lines.connect(self._set_use_straight_lines)
        self.overlay.events.contour_width.connect(self._on_width_change)
        layer.events.selected_label.connect(self._update_color)
        layer.events.colormap.connect(self._update_color)
        layer.events.opacity.connect(self._update_color)
        layer.bind_key("Escape", self.clear, overwrite=True)
        # set completion radius based on settings

        self.reset()
        self._update_color()
        # If there are no points, it won't be visible
        self.overlay.visible = True
        self.move_mutex = QMutex()
        self.modifiers = None
        self.use_straight_lines = False


    @property
    def shift_down(self):
        return False # self.modifiers & Qt.ShiftModifier if self.modifiers is not None else False

    @property
    def alt_down(self):
        return self.modifiers & Qt.AltModifier if self.modifiers is not None else False

    @staticmethod
    def set_calculator_feature(features, grad_x=None, grad_y=None):
        if features is None:
            VispyMinimalContourOverlay._features_shape = None
            return
        if grad_x is None:
            grad_x = np.empty((0, 0))
        if grad_y is None:
            grad_y = np.empty((0, 0))
        if grad_x.shape != grad_y.shape or (grad_x.size != 0 and grad_x.shape != features.shape[:2]):
            raise ValueError("feature and gradient shapes should be equal")
        if features.ndim == 2:
            features = np.tile(features[..., None],(1, 1, 3))
        VispyMinimalContourOverlay.mc_calculator.set_image(features, grad_x, grad_y)
        VispyMinimalContourOverlay._features_shape = features.shape[:2]

    def _set_use_straight_lines(self):
        self.use_straight_lines = self.overlay.use_straight_lines

    def _on_enabled_change(self):
        if self.overlay.enabled:
            self._on_points_change()

    def _on_width_change(self):
        self._last_anchor_to_current_pos_contour.set_data(width=self.overlay.contour_width)
        self._current_pos_to_first_anchor_contour.set_data(width=self.overlay.contour_width)
        self._stored_contour.set_data(width=self.overlay.contour_width)

    def _on_points_change(self):
        current_pos_to_first_anchor_points = np.round(self.overlay.current_pos_to_first_anchor_contour).reshape(-1, self.layer.ndim)
        stored_contour_points = np.round(self.overlay.stored_contour).reshape(-1, self.layer.ndim)
        last_anchor_to_current_pos_points = np.round(self.overlay.last_anchor_to_current_pos_contour).reshape(-1, self.layer.ndim)
        # points = np.concatenate([current_pos_to_first_anchor_points, stored_contour_points, last_anchor_to_current_pos_points], axis=0)

        anchor_points = self.overlay.anchor_points

        if len(anchor_points):
            anchor_points = np.array(anchor_points)[
                     :, self._dims_displayed[::-1]
                     ]
        else:
            anchor_points = np.empty((0, 2))

        # try:
        for visual, new_data, is_dashed in [
            (self._stored_contour, stored_contour_points, False),
            (self._last_anchor_to_current_pos_contour, last_anchor_to_current_pos_points, True),
            (self._current_pos_to_first_anchor_contour, current_pos_to_first_anchor_points, True)
        ]:
            if len(new_data):
                new_data = np.array(new_data)[
                         :, self._dims_displayed[::-1]
                         ]
            else:
                new_data = np.empty((0, 2))
            if len(new_data) > 2:
                if isinstance(visual, Polygon):
                    visual.pos = new_data
                else:
                    if is_dashed:
                        visual.set_data(new_data, connect=get_dashed_cache(len(new_data)))
                    else:
                        visual.set_data(new_data)
                visual.visible = True
            else:
                visual.visible = False

        # except AssertionError:
        #     ...
        # except ValueError:
        #     ...
        # except IndexError:
        #     ...
        # except Exception:
        #     traceback.print_exc()
        self._anchor_points.set_data(
            pos=anchor_points,
            **self._nodes_kwargs,
        )

    def _set_color(self, color):
        border_color = tuple(color[:3]) + (1,)  # always opaque
        polygon_color = color

        # Clean up polygon faces before making it transparent, otherwise
        # it keeps the previous visualization of the polygon without cleaning
        # if polygon_color[-1] == 0:
        #     self._polygon.mesh.set_data(faces=[])
        # self._polygon.color = polygon_color

        # self._polygon.border_color = border_color
        self._stored_contour.set_data(color=border_color)

    def _update_color(self):
        layer = self.layer
        if layer.selected_label == layer.colormap.background_value:
            self._set_color((1, 0, 0, 0))
        else:
            self._set_color(
                layer._selected_color.tolist()[:3] + [layer.opacity]
            )

    @_only_when_enabled
    def _on_mouse_move(self, layer, event):
        """Continuously redraw the latest polygon point with the current mouse position."""
        if not self.move_mutex.tryLock():
            return
        try:
            if self._num_points == 0:
                return
            # image_layer = self.image_layer
            # if image_layer is None or layer._slice_input.ndisplay == 3:
            #     return
            dims_displayed = self._dims_displayed
            pos = self._get_mouse_coordinates(event)
            pos = np.clip(pos, 0, np.subtract(self.layer.data.shape, 1))
            self.overlay.anchor_points = self.overlay.anchor_points[:-1] + [pos.tolist()]
            self.point_triangle[1] = np.asarray(pos)[dims_displayed]
            if np.any(self.point_triangle < 0):
                return
            if self.use_straight_lines:
                results = [
                    np.asarray(skimage.draw.line(
                        round(self.point_triangle[1, 0]), round(self.point_triangle[1, 1]),
                        round(self.point_triangle[0, 0]), round(self.point_triangle[0, 1])
                    )).T,
                    np.asarray(skimage.draw.line(
                        round(self.point_triangle[2, 0]), round(self.point_triangle[2, 1]),
                        round(self.point_triangle[1, 0]), round(self.point_triangle[1, 1])
                    )).T
                ]
            else:
                if self._features_shape is None:
                    warnings.warn("Features not initialized.")
                    return
                results = self._estimate()

            new_e_data = np.tile(np.where([i in dims_displayed for i in range(self.layer.ndim)], np.nan,
                                          self.overlay.anchor_points[0]), [len(results[0]), 1])
            new_s_data = np.tile(np.where([i in dims_displayed for i in range(self.layer.ndim)], np.nan,
                                          self.overlay.anchor_points[0]), [len(results[1]), 1])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                order = layer_get_order(self.layer)
                new_e_data[np.isnan(new_e_data)] = np.flipud(results[0][:, order[:2]]).reshape(-1)
                new_s_data[np.isnan(new_s_data)] = np.flipud(results[1][:, order[:2]]).reshape(-1)

            self.overlay.last_anchor_to_current_pos_contour = new_e_data.tolist()
            self.overlay.current_pos_to_first_anchor_contour = new_s_data.tolist()

        finally:
            self.move_mutex.unlock()

    @_only_when_enabled
    def _on_mouse_press(self, layer, event):
        pos = self._get_mouse_coordinates(event)
        dims_displayed = self._dims_displayed
        pos[dims_displayed] = np.clip(pos[dims_displayed], 0,
                                      np.subtract(self.layer.data.shape, 1)[dims_displayed])
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged:
            return

        if event.button == 1:  # left mouse click

            orig_pos = pos.copy()
            # recenter the point in the center of the image pixel
            pos[dims_displayed] = np.round(pos[dims_displayed])
            #TODO handle undo

            prev_point = (
                None if self._num_points <= 1 else self.overlay.anchor_points[0] if self.shift_down else self.overlay.anchor_points[-2]
            )
            # Add a new point only if it differs from the previous one
            if prev_point is None or not np.allclose(np.round(prev_point), np.round(pos)):
                if self.shift_down:
                    self.overlay.anchor_points.insert(0, pos.tolist())
                    if len(self.overlay.current_pos_to_first_anchor_contour):
                        self.overlay.stored_contour = (self.overlay.current_pos_to_first_anchor_contour) + self.overlay.stored_contour
                        self.overlay.current_pos_to_first_anchor_contour = []
                else:
                    self.overlay.anchor_points = self.overlay.anchor_points[:-1] + [pos.tolist()]
                    if len(self.overlay.last_anchor_to_current_pos_contour):
                        self.overlay.stored_contour.extend(self.overlay.last_anchor_to_current_pos_contour)
                        self.overlay.last_anchor_to_current_pos_contour = []
                self.overlay.anchor_points.append((orig_pos + 1e-3).tolist())
                self.point_triangle[-1] = [self.overlay.anchor_points[0][d] for d in dims_displayed]
                self.point_triangle[0] = [self.overlay.anchor_points[-1][d] for d in dims_displayed]
        # TODO handle undo
        # elif event.button == 2 and self._num_points > 0:  # right mouse click
        #     if self._num_points < 3:
        #         self.overlay.anchor_points = []
        #     else:
        #         self.overlay.anchor_points = self.overlay.anchor_points[:-2] + [pos.tolist()]

    @_only_when_enabled
    def _on_mouse_double_click(self, layer, event):
        if event.button == 2:
            self._on_mouse_press(layer, event)
            return None
        self.overlay.stored_contour = self.overlay.current_pos_to_first_anchor_contour + self.overlay.stored_contour
        # Remove the last point from double click, but keep the vertex
        self.overlay.anchor_points = self.overlay.anchor_points[:-1]

        self.overlay.add_polygon_to_labels(layer)
        return None

    def _get_mouse_coordinates(self, event):
        pos = mouse_event_to_labels_coordinate(self.layer, event)
        if pos is None:
            return None

        pos = np.array(pos, dtype=float)
        # pos = np.floor(pos)+0.5
        # pos[self._dims_displayed] += 0.5

        return pos

    @property
    def _dims_displayed(self):
        return self.layer._slice_input.displayed

    @property
    def _num_points(self):
        return len(self.overlay.anchor_points)

    def reset(self):
        super().reset()
        self._on_points_change()

    def _estimate(self):
        from_i, to_i = bbox_around_points(self.point_triangle)
        from_i = np.clip(from_i, 0, np.asarray(self._features_shape)).astype(int)
        to_i = np.clip(to_i, 0, np.asarray(self._features_shape)).astype(int)
        self.mc_calculator.set_boundaries(from_i[1], from_i[0], to_i[1], to_i[0])
        results = self.mc_calculator.run(
            self.point_triangle,
            True,
            True,
            True
        )
        return results

    def clear(self, _):
        self.overlay.last_anchor_to_current_pos_contour = []
        self.overlay.current_pos_to_first_anchor_contour = []
        self.overlay.stored_contour = []
        self.overlay.anchor_points = []
        self.reset()


visual.overlay_to_visual[MinimalContourOverlay] = VispyMinimalContourOverlay