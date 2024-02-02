from packaging import version
import napari
NAPARI_VERSION = version.parse(napari.__version__)
__all__ = ["NAPARI_VERSION"]
