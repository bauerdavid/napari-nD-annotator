import warnings

from qtpy.QtWidgets import QSlider
from qtpy import QtGui
from qtpy.QtCore import Qt
from scipy.ndimage.filters import gaussian_filter
from napari.layers import Image


class BlurSlider(QSlider):
    def __init__(self, viewer, image_layer=None, blur_func=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._smoothed_layer = None
        self.image_layer = image_layer
        self.valueChanged.connect(self.update_image)
        self.setOrientation(Qt.Horizontal)
        self.blur_func = blur_func if blur_func is not None else lambda img, val: gaussian_filter(img, val)

    def setMaximum(self, a0: float) -> None:
        super().setMaximum(a0*10)

    def setMinimum(self, a0: float) -> None:
        super().setMinimum(a0*10)

    def setValue(self, a0: float) -> None:
        super().setValue(a0*10)

    def value(self):
        return super().value()/10

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(ev)
        image_layer = self.image_layer
        if image_layer is None:
            return
        blurred = self.get_blurred_image()
        new_layer_name = "[smooth] %s" % image_layer.name
        if new_layer_name in self.viewer.layers:
            self._smoothed_layer = self.viewer.layers[new_layer_name]
            self._smoothed_layer.data = blurred
            self._smoothed_layer.translate = image_layer.translate[list(self.viewer.dims.displayed)]
            self._smoothed_layer.colormap = image_layer.colormap
            self._smoothed_layer.contrast_limits = image_layer.contrast_limits
            self._smoothed_layer.rgb = image_layer.rgb
        else:
            self._smoothed_layer = Image(
                blurred,
                name="[smooth] %s" % image_layer.name,
                translate=image_layer.translate[list(self.viewer.dims.displayed)],
                colormap=image_layer.colormap,
                contrast_limits=image_layer.contrast_limits,
                rgb=image_layer.rgb
            )
            self.viewer.add_layer(self._smoothed_layer)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(ev)
        if self._smoothed_layer is None:
            return
        self.viewer.layers.remove(self._smoothed_layer)
        self._smoothed_layer = None

    def update_image(self, _):
        if self.image_layer is not None and self._smoothed_layer is not None:
            self._smoothed_layer.data = self.get_blurred_image()
            self._smoothed_layer.events.data()

    def get_blurred_image(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.blur_func(self.image_layer._data_view, self.value())
