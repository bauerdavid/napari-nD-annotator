
__version__ = "0.2.0"

from ._widgets import AnnotatorWidget, InterpolationWidget, MinimalSurfaceWidget
import napari
from packaging import version
if version.parse(napari.__version__) >= version.parse("0.4.15"):
    from ._widgets import ListWidgetBB
    __all__ = ["AnnotatorWidget", "InterpolationWidget", "MinimalSurfaceWidget", "ListWidgetBB"]
else:
    __all__ = ["AnnotatorWidget", "InterpolationWidget", "MinimalSurfaceWidget"]

