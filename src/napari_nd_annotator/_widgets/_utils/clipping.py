"""Viewer-independent helpers for limiting layers to a box in the 3D view with napari clipping planes.

``experimental_clipping_planes`` only act in the 3D canvas, the 2D canvas ignores them.

The planes are given in *world* coordinates (vispy evaluates them in the scene, after the scale/translate of the
layer, even though napari's documentation calls them data coordinates), in the order of the displayed axes. A point
is kept when ``dot(point - position, normal) >= 0``, i.e. the normal points into the kept region.

:class:`LayerClipper` clips layers and shows optional "ghost" copies of them: layers sharing the data array of the
original (no data copy), without clipping planes and with a lower opacity, so that the surroundings of the clipped
region stay visible. Everything created here is tagged with ``metadata[SCAFFOLD_KEY] = True``.
"""
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from napari.layers import Layer

SCAFFOLD_KEY = "nd_annotator_scaffold"
GHOST_OPACITY_FACTOR = 0.25

# properties of the original layer that the ghost follows (the ones a layer does not have are skipped)
_SYNCED_PROPERTIES = (
    "colormap", "contrast_limits", "gamma", "interpolation2d", "interpolation3d", "rendering", "blending",
    "iso_threshold", "attenuation", "contour", "show_selected_label",
    "scale", "translate", "rotate", "shear", "affine", "visible",
)


def is_scaffold(layer) -> bool:
    """Whether the layer was created by the plugin only to support the display of other layers."""
    return bool(getattr(layer, "metadata", None) and layer.metadata.get(SCAFFOLD_KEY, False))


def range_clipping_planes(lower: Sequence[float], upper: Sequence[float]) -> List[dict]:
    """Two planes per axis keeping ``lower[i] <= x[i] <= upper[i]`` (positions along the given axes only)."""
    planes = []
    n = len(lower)
    for i in range(n):
        for bound, sign in ((lower[i], 1), (upper[i], -1)):
            position = [0.] * n
            position[i] = float(bound)
            normal = [0.] * n
            normal[i] = float(sign)
            planes.append({"position": position, "normal": normal})
    return planes


def displayed_layer_dims(viewer, layer) -> List[int]:
    """The dimensions of the layer that are displayed in the viewer, in the order of the display."""
    offset = viewer.dims.ndim - layer.ndim
    return [d - offset for d in viewer.dims.displayed if d >= offset]


def bbox_clipping_planes(viewer, layer, box_min: Sequence[float], box_max: Sequence[float]) -> List[dict]:
    """Clipping planes keeping the voxels ``box_min..box_max`` (both inclusive, data coordinates) of the layer.

    A voxel with integer index ``i`` covers ``[i - .5, i + .5]``, so the planes are placed half a voxel outside.
    Only the displayed dimensions are clipped. The planes are transformed to world coordinates with the scale,
    translate (and rotate/shear/affine) of the layer.
    """
    dims = displayed_layer_dims(viewer, layer)
    if len(dims) != 3:
        return []
    lower = np.asarray(box_min, dtype=float) - .5
    upper = np.asarray(box_max, dtype=float) + .5
    center = (lower + upper) / 2
    origin = np.asarray(layer.data_to_world(np.zeros(layer.ndim)))
    linear = np.stack([np.asarray(layer.data_to_world(axis)) - origin for axis in np.eye(layer.ndim)], axis=1)
    normal_transform = np.linalg.inv(linear[np.ix_(dims, dims)]).T  # normals do not transform like positions
    planes = []
    for k, dim in enumerate(dims):
        for bound, sign in ((lower[dim], 1.), (upper[dim], -1.)):
            point = center.copy()
            point[dim] = bound
            normal = sign * normal_transform[:, k]
            planes.append({"position": [float(v) for v in np.asarray(layer.data_to_world(point))[dims]],
                           "normal": [float(v) for v in normal / np.linalg.norm(normal)]})
    return planes


def _planes_state(layer) -> list:
    return [{"position": tuple(p.position), "normal": tuple(p.normal), "enabled": getattr(p, "enabled", True)}
            for p in layer.experimental_clipping_planes]


class _Ghost:
    """A layer mirroring another one with the same data array, kept in sync with it."""

    def __init__(self, viewer, layer: Layer, opacity_factor: float):
        self.viewer = viewer
        self.layer = layer
        self.opacity_factor = opacity_factor
        data, state, type_str = layer.as_layer_data_tuple()  # `data` is the array of the layer, not a copy
        state["name"] = "%s (surroundings)" % layer.name
        state["metadata"] = {SCAFFOLD_KEY: True}
        state["opacity"] = layer.opacity * opacity_factor
        state["experimental_clipping_planes"] = []
        self.ghost = Layer.create(data, state, type_str)
        self.ghost.editable = False
        self._callbacks = []
        selection = self.viewer.layers.selection
        previous, active = set(selection), selection.active
        # below the original, so that the clipped region is drawn over its surroundings
        self.viewer.layers.insert(self.viewer.layers.index(layer), self.ghost)
        selection.clear()
        selection.update(previous)
        if active in previous:
            selection.active = active
        self._connect("data", self._on_data)
        self._connect("set_data", self._on_refresh)
        self._connect("opacity", self._on_opacity)
        for name in _SYNCED_PROPERTIES:
            self._connect(name, self._property_handler(name))
        self._sync_visibility()

    def _connect(self, event_name, callback):
        emitter = getattr(self.layer.events, event_name, None)
        if emitter is not None:
            emitter.connect(callback)
            self._callbacks.append((emitter, callback))

    def _property_handler(self, name):
        def handler(_=None):
            try:
                setattr(self.ghost, name, getattr(self.layer, name))
            except (AttributeError, ValueError, TypeError):
                pass
        return handler

    def _on_data(self, _=None):
        self.ghost.data = self.layer.data

    def _on_refresh(self, _=None):
        # painting changes the (shared) array in place
        self.ghost.refresh()

    def _on_opacity(self, _=None):
        self.ghost.opacity = self.layer.opacity * self.opacity_factor

    def _sync_visibility(self):
        self.ghost.visible = self.layer.visible

    def close(self):
        for emitter, callback in self._callbacks:
            try:
                emitter.disconnect(callback)
            except (ValueError, TypeError, RuntimeError):
                pass
        self._callbacks = []
        if self.ghost in self.viewer.layers:
            self.viewer.layers.remove(self.ghost)


class LayerClipper:
    """Limits layers to a box in the 3D view and optionally shows their surroundings as ghosts.

    ``release()`` restores the clipping planes and the visibility the layers had before they were clipped and
    removes all ghosts. It also happens automatically when a clipped layer leaves the viewer.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self._saved: Dict[Layer, dict] = dict()
        self._ghosts: Dict[Layer, _Ghost] = dict()
        self._box = None
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.dims.events.order.connect(self._apply_planes)
        self.viewer.dims.events.ndisplay.connect(self._apply_planes)

    @property
    def layers(self) -> List[Layer]:
        return list(self._saved)

    @property
    def ghosts(self) -> List[Layer]:
        return [ghost.ghost for ghost in self._ghosts.values()]

    def clip(self, layers: Iterable[Optional[Layer]], box_min, box_max, make_visible=True):
        """Starts clipping the layers to the box (data coordinates, both ends inclusive)."""
        for layer in layers:
            if layer is None:
                continue
            if layer not in self._saved:
                self._saved[layer] = {"planes": _planes_state(layer), "visible": layer.visible}
                for emitter in self._transform_emitters(layer):
                    emitter.connect(self._apply_planes)
            # setting `visible = True` on a visible layer would reset the mode of a labels layer
            if make_visible and not layer.visible:
                layer.visible = True
        self.set_box(box_min, box_max)

    @staticmethod
    def _transform_emitters(layer):
        return [getattr(layer.events, name) for name in ("scale", "translate") if hasattr(layer.events, name)]

    def set_box(self, box_min, box_max):
        self._box = (np.asarray(box_min, dtype=float), np.asarray(box_max, dtype=float))
        self._apply_planes()

    def _apply_planes(self, _=None):
        if self._box is None:
            return
        for layer in self._saved:
            layer.experimental_clipping_planes = bbox_clipping_planes(self.viewer, layer, *self._box)

    def set_ghost(self, layer: Optional[Layer], enabled: bool, opacity_factor=GHOST_OPACITY_FACTOR):
        """Shows or hides the unclipped ghost of a layer that is being clipped."""
        if layer is None or layer not in self._saved:
            return
        if enabled and layer not in self._ghosts:
            self._ghosts[layer] = _Ghost(self.viewer, layer, opacity_factor)
        elif not enabled and layer in self._ghosts:
            self._ghosts.pop(layer).close()

    def _on_layer_removed(self, event):
        layer = event.value
        if layer in self._ghosts:
            self._ghosts.pop(layer).close()
        for original, ghost in list(self._ghosts.items()):
            if ghost.ghost is layer:  # closed by hand
                self._ghosts.pop(original).close()
        if self._saved.pop(layer, None) is not None and not self._saved:
            self.release()

    def release(self):
        """Restores the layers and removes the ghosts. The clipper can be reused afterwards."""
        for ghost in list(self._ghosts.values()):
            ghost.close()
        self._ghosts.clear()
        for layer, state in list(self._saved.items()):
            for emitter in self._transform_emitters(layer):
                try:
                    emitter.disconnect(self._apply_planes)
                except (ValueError, TypeError, RuntimeError):
                    pass
            if layer in self.viewer.layers:
                layer.experimental_clipping_planes = state["planes"]
                if layer.visible != state["visible"]:
                    layer.visible = state["visible"]
        self._saved.clear()
        self._box = None

    def close(self):
        """Releases everything and disconnects from the viewer."""
        self.release()
        for emitter, callback in ((self.viewer.layers.events.removed, self._on_layer_removed),
                                  (self.viewer.dims.events.order, self._apply_planes),
                                  (self.viewer.dims.events.ndisplay, self._apply_planes)):
            try:
                emitter.disconnect(callback)
            except (ValueError, TypeError, RuntimeError):
                pass
