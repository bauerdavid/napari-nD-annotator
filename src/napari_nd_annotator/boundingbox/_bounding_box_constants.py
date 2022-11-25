# A copy of napari.layers.shapes._shapes_constants
import sys
from enum import auto

from napari.utils.misc import StringEnum


class Mode(StringEnum):
    """Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    The SELECT mode allows for bounding boxes to be selected, moved and
    resized.

    The ADD_BOUNDING_BOX mode allows for bounding boxes to be added.

    The DIRECT mode is unused, and will be removed
    """

    PAN_ZOOM = auto()
    SELECT = auto()
    DIRECT = auto()
    ADD_BOUNDING_BOX = auto()



class ColorMode(StringEnum):
    """
    ColorMode: Color setting mode.

    DIRECT (default mode) allows each shape to be set arbitrarily

    CYCLE allows the color to be set via a color cycle over an attribute

    COLORMAP allows color to be set via a color map over an attribute
    """

    DIRECT = auto()
    CYCLE = auto()
    COLORMAP = auto()


class SizeMode(StringEnum):
    AVERAGE = auto()
    CONSTANT = auto()


class Box:
    """Box: Constants associated with the vertices of the interaction box"""

    WITH_HANDLE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    WITHOUT_HANDLE = list(range(8))
    LINE_HANDLE = [7, 6, 4, 2, 0, 7]
    LINE = [0, 2, 4, 6, 0]
    TOP_LEFT = 0
    TOP_CENTER = 7
    LEFT_CENTER = 1
    BOTTOM_RIGHT = 4
    BOTTOM_LEFT = 2
    CENTER = 8
    HANDLE = 9
    LEN = 8


BACKSPACE = 'delete' if sys.platform == 'darwin' else 'backspace'
