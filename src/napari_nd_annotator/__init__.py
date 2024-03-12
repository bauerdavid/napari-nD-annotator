
__version__ = "0.2.3"

from ._widgets import AnnotatorWidget, InterpolationWidget, MinimalSurfaceWidget
from packaging import version
from ._napari_version import NAPARI_VERSION

if NAPARI_VERSION >= version.parse("0.4.15"):
    from ._widgets import ListWidgetBB
    __all__ = ["AnnotatorWidget", "InterpolationWidget", "MinimalSurfaceWidget", "ListWidgetBB", "NAPARI_VERSION"]
else:
    __all__ = ["AnnotatorWidget", "InterpolationWidget", "MinimalSurfaceWidget", "NAPARI_VERSION"]

