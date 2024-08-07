import napari
import numpy as np
from napari import layers
from packaging import version
import warnings

from ._napari_version import NAPARI_VERSION

if NAPARI_VERSION >= version.parse("0.4.18"):
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
else:
    def layer_dims_displayed(layer: layers.Layer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return layer._dims_displayed

    def layer_dims_not_displayed(layer: layers.Layer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return layer._dims_not_displayed

    def layer_ndisplay(layer: layers.Layer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return layer._ndisplay


if NAPARI_VERSION >= version.parse("0.5.0"):
    from napari.utils.misc import reorder_after_dim_reduction


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
else:
    def layer_slice_indices(layer: layers.Layer):
        return layer._slice_indices

    def layer_get_order(layer: layers.Layer):
        return layer._get_order()