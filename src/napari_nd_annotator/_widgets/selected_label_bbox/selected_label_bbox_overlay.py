try:
    from napari._pydantic_compat import Field
except ImportError:
    from pydantic import Field
from napari.components.overlays import SceneOverlay
from napari.utils.color import ColorValue

DEFAULT_BBOX_COLOR = "yellow"


class SelectedLabelBBoxOverlay(SceneOverlay):
    """Box around the currently selected label of a Labels layer.

    The box follows the selected label of the layer it belongs to. It is the
    label's bounding box in the whole layer, projected onto the displayed
    dimensions: in 2D it frames the label on all slices, not only the current
    one. It is shown on the slices between the first and last one containing
    the label, and hidden while the background label is selected or the label
    does not occur in the layer.

    Attributes
    ----------
    visible : bool
        Whether the box is shown (off by default).
    color : ColorValue
        Color of the box edges.
    line_width : float
        Width of the box edges in canvas pixels.
    """

    visible: bool = False
    color: ColorValue = Field(default_factory=lambda: ColorValue(DEFAULT_BBOX_COLOR))
    line_width: float = 2.
