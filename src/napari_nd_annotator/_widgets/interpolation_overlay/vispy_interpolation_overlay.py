import numpy as np
from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari._vispy.utils import visual
from napari.layers import Labels
from vispy.scene import Polygon, Compound

from .interpolation_overlay import InterpolationOverlay


class VispyInterpolationOverlay(LayerOverlayMixin, VispySceneOverlay):
    layer: Labels

    def __init__(
        self, *, layer: Labels, overlay: InterpolationOverlay, parent=None
    ):
        points = [(0, 0), (1, 1)]

        self._polygon = Polygon(
            pos=points,
            border_method='gl',
        )

        super().__init__(
            node=Compound([self._polygon]),
            layer=layer,
            overlay=overlay,
            parent=parent,
        )

        self.overlay.events.points_per_slice.connect(self._on_points_change)
        self.overlay.events.current_slice.connect(self._on_points_change)
        self.overlay.events.enabled.connect(self._on_enabled_change)

        layer.events.selected_label.connect(self._update_color)
        layer.events.colormap.connect(self._update_color)
        layer.events.opacity.connect(self._update_color)


        self.reset()
        self._update_color()
        # If there are no points, it won't be visible
        self.overlay.visible = True

    def _on_enabled_change(self):
        if self.overlay.enabled:
            self._on_points_change()

    def _on_points_change(self):
        if self.overlay.current_slice >= len(self.overlay.points_per_slice):
            return
        points = self.overlay.points_per_slice[self.overlay.current_slice]

        if points:
            self._polygon.visible = True
            self._polygon.pos = points
        else:
            self._polygon.visible = False

    def _set_color(self, color):
        border_color = tuple(color[:3]) + (1,)  # always opaque
        polygon_color = color

        # Clean up polygon faces before making it transparent, otherwise
        # it keeps the previous visualization of the polygon without cleaning
        if polygon_color[-1] == 0:
            self._polygon.mesh.set_data(faces=[])
        self._polygon.color = polygon_color

        self._polygon.border_color = border_color

    def _update_color(self):
        layer = self.layer
        if layer._selected_label == layer.colormap.background_value:
            self._set_color((1, 0, 0, 0))
        else:
            self._set_color(
                layer._selected_color.tolist()[:3] + [layer.opacity]
            )

    @property
    def _dims_displayed(self):
        return self.layer._slice_input.displayed

    def reset(self):
        super().reset()
        self._on_points_change()

visual.overlay_to_visual[InterpolationOverlay] = VispyInterpolationOverlay
