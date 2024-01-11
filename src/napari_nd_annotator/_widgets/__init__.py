from .interpolation_widget import InterpolationWidget
from .projections import SliceDisplayWidget
from .annotator_module import AnnotatorWidget
from .minimal_surface_widget import MinimalSurfaceWidget

import napari
from packaging import version
if version.parse(napari.__version__) >= version.parse("0.4.15"):
    from .object_list import ListWidgetBB
    __all__ = ["InterpolationWidget", "SliceDisplayWidget", "AnnotatorWidget", "MinimalSurfaceWidget", "ListWidgetBB"]
else:
    if version.parse(napari.__version__) <= version.parse("0.4.12"):
        from napari.layers import Layer
        def data_to_world(self, position):
            """Convert from data coordinates to world coordinates.
            Parameters
            ----------
            position : tuple, list, 1D array
                Position in data coordinates. If longer then the
                number of dimensions of the layer, the later
                dimensions will be used.
            Returns
            -------
            tuple
                Position in world coordinates.
            """
            if len(position) >= self.ndim:
                coords = list(position[-self.ndim:])
            else:
                coords = [0] * (self.ndim - len(position)) + list(position)

            return tuple(self._transforms[1:].simplified(coords))
        Layer.data_to_world = data_to_world

    __all__ = ["InterpolationWidget", "SliceDisplayWidget", "AnnotatorWidget", "MinimalSurfaceWidget"]
