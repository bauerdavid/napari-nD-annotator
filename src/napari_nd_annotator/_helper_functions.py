import napari
from napari import layers
from packaging import version
import warnings
if version.parse(napari.__version__) >= version.parse("0.4.18"):
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