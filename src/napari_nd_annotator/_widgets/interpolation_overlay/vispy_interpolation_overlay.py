import threading
import time
import traceback
from functools import wraps

import numpy as np
from magicgui._util import debounce
from napari.qt.threading import thread_worker
from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari._vispy.utils import visual
from napari.layers import Labels
from vispy.scene import Polygon, Compound

from .interpolation_overlay import InterpolationOverlay, mutex as overlay_mutex


def execute_last(function=None, delay=0.2):

    def decorator(fn):
        from threading import Timer

        _store: dict = {"retry_worker": None, "mutex": threading.Lock()}

        @wraps(fn)
        def delayed(*args, **kwargs):
            mutex = _store["mutex"]
            def call_it(*args, **kwargs):
                _store["retry_worker"] = None
                with mutex:
                    print("calling function", flush=True)
                    fn(*args, **kwargs)
                    print("called function", flush=True)

            @thread_worker
            def retry(*args, **kwargs):
                while True:
                    if not mutex.locked():
                        call_it(*args, **kwargs)
                        return
                    time.sleep(delay)
                    yield
            if _store["retry_worker"] is not None:
                _store["retry_worker"].quit()
            _store["retry_worker"] = retry(*args, **kwargs)
            _store["retry_worker"].start()
            return None

        return delayed

    return decorator if function is None else decorator(function)


class VispyInterpolationOverlay(LayerOverlayMixin, VispySceneOverlay):
    layer: Labels

    def __init__(
        self, *, layer: Labels, overlay: InterpolationOverlay, parent=None
    ):
        points = np.asarray([(0, 0), (1, 1)])
        self._polygons = [Polygon(
            pos=points,
            border_method='gl',
            border_width=2
        ) for _ in range(1)]

        super().__init__(
            node=Compound(self._polygons),
            layer=layer,
            overlay=overlay,
            parent=parent,
        )
        # self.overlay.events.points_per_slice.connect(self._on_points_change)
        self.overlay.events.contour.connect(self._on_points_change)
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

    # @execute_last
    def _on_points_change(self):
        print("on_points_change")
        # print("initial checks in points change:", time.time() - start, flush=True)
        contours = self.overlay.contour.copy()
        n_contours = len(contours)
        n_poly = len(self._polygons)
        if n_poly < n_contours:
            self._polygons.extend([Polygon(
                pos=np.asarray([(0, 0), (1, 1)]),
                border_method='agg',
                border_width=3
            ) for _ in range(n_contours - n_poly)])
        # print(f"extending polygons: {time.time() - start} (n_poly: {n_poly}, n_contours: {n_contours})", flush=True)
        for i, poly in enumerate(self._polygons):
            points = contours[i] if i < n_contours else None
            if points is not None and len(points) > 2:
                poly.visible = True
                poly.pos = points
            else:
                print("setting poly invisible", flush=True)
                poly.visible = False

    def _set_color(self, color):
        print("set_color")
        border_color = tuple(color[:3]) + (1,)  # always opaque
        polygon_color = color

        # Clean up polygon faces before making it transparent, otherwise
        # it keeps the previous visualization of the polygon without cleaning
        for poly in self._polygons:
            if polygon_color[-1] == 0:
                poly.mesh.set_data(faces=[])
            poly.color = polygon_color

            poly.border_color = border_color

    def _update_color(self):
        print("_update_color")
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
        print("reset")
        super().reset()
        self._on_points_change()

visual.overlay_to_visual[InterpolationOverlay] = VispyInterpolationOverlay
