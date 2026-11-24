from typing import Optional

import numpy as np
from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari._vispy.utils import visual
from napari.layers import Labels
from napari.utils.events import disconnect_events
from qtpy.QtCore import QTimer
from vispy.scene.visuals import Line

from .selected_label_bbox_overlay import SelectedLabelBBoxOverlay

# Corners are generated with itertools.product order over (x, y[, z]) bounds;
# these index pairs are the edges of the rectangle / box.
_EDGES_2D = np.array([[0, 1], [1, 3], [3, 2], [2, 0]])
_EDGES_3D = np.array([
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
])

# while painting or scrolling, the bounds are recomputed once the changes pause for this long
_PAINT_UPDATE_DELAY_MS = 100


def label_bounds(data, label) -> Optional[np.ndarray]:
    """Bounding box of ``label`` in ``data`` as an ``(ndim, 2)`` array of inclusive pixel indices.

    Returns None when the label does not occur in the data.
    """
    mask = np.asarray(data) == label
    if mask.ndim == 0:
        return None
    bounds = []
    while mask.ndim > 0:
        present = np.flatnonzero(mask.any(axis=tuple(range(1, mask.ndim))) if mask.ndim > 1 else mask)
        if present.size == 0:
            return None
        bounds.append((present[0], present[-1]))
        # the remaining axes only need the part of the data where the label occurs along this one
        mask = mask[present[0]:present[-1] + 1].any(axis=0)
    return np.asarray(bounds, dtype=int)


def box_vertices(bounds: np.ndarray) -> np.ndarray:
    """Corners of the box covering the pixels in ``bounds`` (numpy axis order), in vispy (x, y[, z]) order.

    Pixel centers are at integer coordinates, so the box extends half a pixel
    beyond the outermost pixel centers.
    """
    edges = bounds.astype(float) + [-.5, .5]
    edges = edges[::-1]  # numpy -> vispy axis order
    grids = np.meshgrid(*edges, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)


class VispySelectedLabelBBoxOverlay(LayerOverlayMixin, VispySceneOverlay):
    """Draws the bounding box of the selected label over the whole layer, projected onto the displayed dimensions.

    In 2D the box shows the label's extent in the displayed plane across all slices, in 3D
    across all non-displayed dimensions (e.g. time). It is shown only while the current
    position is within the label's range along every non-displayed dimension: with the label
    annotated on slices 3 and 15, the box is drawn on slices 3 to 15.
    The label's bounds are computed from the full data and cached; changing the view only
    re-projects them.
    """
    layer: Labels

    def __init__(self, *, layer: Labels, overlay: SelectedLabelBBoxOverlay, parent=None, **kwargs):
        self._bounds = None  # (ndim, 2) bounds of the selected label in the whole data, or None
        self._has_box = False
        super().__init__(
            node=Line(pos=np.zeros((2, 3)), connect="segments", method="gl", antialias=True),
            layer=layer,
            overlay=overlay,
            parent=parent,
            **kwargs
        )
        self._recompute_timer = QTimer()
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(_PAINT_UPDATE_DELAY_MS)
        self._recompute_timer.timeout.connect(self._recompute)

        self.overlay.events.visible.connect(self._recompute)
        self.overlay.events.color.connect(self._redraw)
        self.overlay.events.line_width.connect(self._redraw)

        layer.events.selected_label.connect(self._recompute)
        layer.events.data.connect(self._recompute)
        layer.events.paint.connect(self._recompute)  # end of a stroke, undo, redo
        layer.events.visible.connect(self._recompute)
        # emitted for every brush dab while painting
        layer.events.labels_update.connect(self._schedule_recompute)
        # after (re)slicing: the displayed dimensions may have changed; the data may have
        # been written without a paint event (e.g. by the plugin's tools) and then refreshed
        layer.events.set_data.connect(self._on_set_data)

        self.reset()

    def _should_be_visible(self) -> bool:
        return bool(self.overlay.visible and self.layer.visible and self._has_box)

    def _on_visible_change(self):
        self.node.visible = self._should_be_visible()

    def _schedule_recompute(self, *_):
        if self.overlay.visible:
            self._recompute_timer.start()

    def _on_set_data(self, *_):
        self._redraw()
        self._schedule_recompute()

    def _label_bounds(self) -> Optional[np.ndarray]:
        layer = self.layer
        if layer.multiscale:
            # the overlay lives in the coordinates of the displayed tile; not supported
            return None
        selected_label = layer.selected_label
        if selected_label == layer.colormap.background_value:
            return None
        return label_bounds(layer.data, selected_label)

    def _recompute(self, *_):
        self._recompute_timer.stop()
        if self.overlay.visible and self.layer.visible:
            self._bounds = self._label_bounds()
        else:
            # not kept up to date while hidden; recomputed when shown again
            self._bounds = None
        self._redraw()

    def _redraw(self, *_):
        bounds = self._bounds
        displayed = list(self.layer._slice_input.displayed)
        self._has_box = (bounds is not None and len(bounds) == self.layer.ndim
                         and self._current_position_in(bounds))
        if self._has_box:
            # numpy order of the displayed dimensions, as the layer's node expects
            vertices = box_vertices(bounds[displayed])
            if len(displayed) == 2:
                # a flat box at depth 0, like napari draws 2D layers in 3D
                vertices = np.pad(vertices, ((0, 0), (0, 1)))
                edges = _EDGES_2D
            else:
                edges = _EDGES_3D
            self.node.set_data(
                pos=vertices,
                connect=edges.copy(),
                color=np.asarray(self.overlay.color),
                width=self.overlay.line_width,
            )
        self._on_visible_change()

    def _current_position_in(self, bounds: np.ndarray) -> bool:
        """Whether the displayed slice lies within ``bounds`` along every non-displayed dimension."""
        not_displayed = list(self.layer._slice_input.not_displayed)
        if not not_displayed:
            return True
        # rounded the way napari picks the slice to show
        position = np.round(np.asarray(self.layer._data_slice.point, dtype=float)[not_displayed])
        bounds = bounds[not_displayed]
        return bool(np.all((bounds[:, 0] <= position) & (position <= bounds[:, 1])))

    def reset(self):
        super().reset()
        self._recompute()

    def close(self):
        self._recompute_timer.stop()
        disconnect_events(self.layer.events, self)
        super().close()


visual.overlay_to_visual[SelectedLabelBBoxOverlay] = VispySelectedLabelBBoxOverlay
