
__version__ = "0.0.8"

from ._widgets import AnnotatorWidget, InterpolationWidget
import napari
from packaging import version
if version.parse(napari.__version__) >= version.parse("0.4.15"):
    from ._widgets import ListWidgetBB
    __all__ = ["AnnotatorWidget", "InterpolationWidget", "ListWidgetBB"]
else:
    __all__ = ["AnnotatorWidget", "InterpolationWidget"]

