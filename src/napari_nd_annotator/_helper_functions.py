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
    """The index of the displayed slice along each dimension that is not displayed (``slice(None)`` for the others).

    Rounded to the nearest index like napari does when it picks the slice to show: the data position of the slice is
    fractional when the world position does not fall on a voxel of the layer (another layer with a finer z step,
    a translate that is not a multiple of the scale, or just floating point error, e.g. 3.9999 for slice 4).
    """
    return tuple(slice(None) if np.isnan(p) else int(np.round(p)) for p in layer._data_slice.point)


# --- the transform of a layer (data -> world): napari composes it from these, in this order
TRANSFORM_COMPONENTS = ("scale", "translate", "rotate", "shear", "affine")

_transform_sync = {"depth": 0}


class syncing_transforms:
    """Context in which the plugin copies transforms between layers: the transform events this emits are not user
    changes to be passed on again (see `is_syncing_transforms`)."""

    def __enter__(self):
        _transform_sync["depth"] += 1
        return self

    def __exit__(self, *exc):
        _transform_sync["depth"] -= 1
        return False


def is_syncing_transforms():
    return _transform_sync["depth"] > 0


def layer_transform(layer: layers.Layer, offset=None, factor=None) -> dict:
    """The transform components of ``layer`` (as keyword arguments of a napari layer).

    With ``offset`` and/or ``factor`` it is the transform of a layer whose voxel ``i`` is voxel
    ``offset + factor * i`` of ``layer`` (a crop, or a crop resampled along some axes), so that the two show the
    same place in the world.
    """
    from napari.utils.transforms.transform_utils import shear_to_matrix
    ndim = layer.ndim
    scale = np.asarray(layer.scale, dtype=float).copy()
    translate = np.asarray(layer.translate, dtype=float).copy()
    rotate = np.asarray(layer.rotate, dtype=float).copy()
    shear = np.asarray(shear_to_matrix(np.asarray(layer.shear, dtype=float), ndim=ndim), dtype=float)
    if offset is not None:
        # napari: data -> physical is translate + rotate @ shear @ diag(scale) @ index
        linear = rotate @ shear @ np.diag(scale)
        translate = translate + linear @ np.asarray(offset, dtype=float)
    if factor is not None:
        scale = scale * np.asarray(factor, dtype=float)
    return {"scale": scale, "translate": translate, "rotate": rotate, "shear": shear,
            "affine": np.asarray(layer.affine.affine_matrix, dtype=float).copy()}


def copy_transform(reference: layers.Layer, layer: layers.Layer) -> list:
    """Gives ``layer`` the transform of ``reference`` (its data are left alone: voxel indices of the two then mean
    the same place). Layers of different dimensionality share the transform of their last dimensions.

    Returns the names of the components that changed.
    """
    source, target = layer_transform(reference), layer_transform(layer)
    count = min(reference.ndim, layer.ndim)
    new = {name: value.copy() for name, value in target.items()}
    for name in ("scale", "translate"):
        new[name][-count:] = source[name][-count:]
    for name in ("rotate", "shear"):
        new[name][-count:, -count:] = source[name][-count:, -count:]
    # the affine matrix is (ndim + 1) x (ndim + 1), its last column is the translation
    new["affine"][-count - 1:-1, -count - 1:-1] = source["affine"][-count - 1:-1, -count - 1:-1]
    new["affine"][-count - 1:-1, -1] = source["affine"][-count - 1:-1, -1]
    changed = [name for name in TRANSFORM_COMPONENTS if not np.allclose(new[name], target[name])]
    with syncing_transforms():
        for name in changed:
            setattr(layer, name, new[name])
    return changed


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


def unwrap_viewer(viewer):
    """The real viewer behind napari's ``PublicOnlyProxy``.

    napari hands plugin widgets (Plugins menu, ``add_plugin_dock_widget``) a proxied
    viewer. Everything reached through it is proxied too, and Qt rejects a proxied
    widget (e.g. as a parent: "unexpected type 'PublicOnlyProxy'"), while private
    attributes read through it warn. The widgets keep the unwrapped viewer instead.
    """
    while viewer is not None and type(viewer).__name__.endswith("PublicOnlyProxy"):
        viewer = viewer.__wrapped__
    return viewer


def viewer_qt_viewer(viewer):
    """The ``QtViewer`` of the viewer's window.

    napari deprecated the public ``Window.qt_viewer`` (to be removed in 0.10 at the
    earliest) but kept it as the private ``Window._qt_viewer``, which older versions
    don't have.
    """
    window = unwrap_viewer(viewer).window
    if hasattr(window, "_qt_viewer"):
        return window._qt_viewer
    return window.qt_viewer


def viewer_main_window(viewer):
    """The viewer's Qt main window, found from its ``QtViewer`` with Qt's own
    ``QWidget.window()`` instead of the private ``Window._qt_window``."""
    return viewer_qt_viewer(viewer).window()
