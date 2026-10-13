import napari
import numpy as np
from napari import layers
from napari.utils.misc import reorder_after_dim_reduction
import warnings

try:
    from napari.layers.labels.labels import _coerce_indices_for_vectorization
except ImportError:
    import numpy.typing as npt
    import inspect


    def _arraylike_short_names(obj):
        """Yield all the short names of an array-like or its class."""
        type_ = type(obj) if not inspect.isclass(obj) else obj
        for base in type_.mro():
            yield f'{base.__module__.split(".", maxsplit=1)[0]}.{base.__name__}'


    def _is_array_type(array: npt.ArrayLike, type_name: str) -> bool:
        return type_name in _arraylike_short_names(array)


    def _coerce_indices_for_vectorization(array, indices: list) -> tuple:
        """Coerces indices so that they can be used for vectorized indexing in the given data array."""
        if _is_array_type(array, 'xarray.DataArray'):
            # Fix indexing for xarray if necessary
            # See http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
            # for difference from indexing numpy
            try:
                import xarray as xr
            except ModuleNotFoundError:
                pass
            else:
                return tuple(xr.DataArray(i) for i in indices)
        return tuple(indices)


def layer_dims_displayed(layer: layers.Layer):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return layer._slice_input.displayed


def layer_dims_not_displayed(layer: layers.Layer):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return layer._slice_input.not_displayed


def layer_ndisplay(layer: layers.Layer):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return layer._slice_input.ndisplay


def layer_dims_order(layer: layers.Layer):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return layer._slice_input.order


def layer_slice_indices(layer: layers.Layer):
    return tuple(slice(None) if np.isnan(p) else int(p) for p in layer._data_slice.point)


def layer_get_order(layer: layers.Layer):
    order = reorder_after_dim_reduction(layer._slice_input.displayed)
    if len(layer.data.shape) != layer.ndim:
        # if rgb need to keep the final axis fixed during the
        # transpose. The index of the final axis depends on how many
        # axes are displayed.
        return (*order, max(order) + 1)

    return order


def viewer_canvas_size(viewer):
    """Size of the canvas in pixels, as (height, width).

    napari 0.9 removed the private ``Viewer._canvas_size`` in favour of the
    public ``viewer.canvas.size``, which has the same (height, width) order.
    """
    canvas = getattr(viewer, "canvas", None)
    if canvas is not None and hasattr(canvas, "size"):
        return canvas.size
    return viewer._canvas_size


def viewer_camera(viewer):
    """The camera model of the viewer.

    napari 0.9 deprecated ``viewer.camera`` in favour of ``viewer.scene.camera``;
    older versions have no ``viewer.scene``.
    """
    scene = getattr(viewer, "scene", None)
    if scene is not None and hasattr(scene, "camera"):
        return scene.camera
    return viewer.camera


def viewer_qt_viewer(viewer):
    """The ``QtViewer`` of the viewer's window.

    napari deprecated the public ``Window.qt_viewer`` (to be removed in 0.10 at the
    earliest) but kept it as the private ``Window._qt_viewer``, which older versions
    don't have.
    """
    window = viewer.window
    if hasattr(window, "_qt_viewer"):
        return window._qt_viewer
    return window.qt_viewer
