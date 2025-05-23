import numpy as np
from napari._pydantic_compat import Field
from napari.components.overlays import SceneOverlay
import threading
from copy import deepcopy

mutex = threading.Lock()

class ContourList(list):
    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for cnt1, cnt2 in zip(self, other):
            if not np.array_equal(cnt1, cnt2):
                return False
        return True


class InterpolationOverlay(SceneOverlay):
    """Overlay that displays a polygon on a scene.

    This overlay was created for drawing polygons on Labels layers. It handles
    the following mouse events to update the overlay:
    - Mouse move: Continuously redraw the latest polygon point with the current
    mouse position.
    - Mouse press (left button): Adds the current mouse position as a new
    polygon point.
    - Mouse double click (left button): If there are at least three points in
    the polygon and the double-click position is within completion_radius
    from the first vertex, the polygon will be painted in the image using the
    current label.
    - Mouse press (right button): Removes the most recent polygon point from
    the list.

    Attributes
    ----------
    enabled : bool
        Controls whether the overlay is activated.
    points : list
        A list of (x, y) coordinates of the vertices of the polygon.
    use_double_click_completion_radius : bool
        Whether double-click to complete drawing the polygon requires being within
        completion_radius of the first point.
    completion_radius : int | float
        Defines the radius from the first polygon vertex within which
        the drawing process can be completed by a left double-click.
    """

    enabled: bool = False
    contour: list = Field(default_factory=ContourList)

