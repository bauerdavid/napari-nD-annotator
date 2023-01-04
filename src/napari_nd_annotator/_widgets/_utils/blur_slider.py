import warnings

from napari._qt.widgets._slider_compat import QDoubleSlider
from qtpy import QtGui
from qtpy.QtCore import Qt, QEvent
from scipy.ndimage.filters import gaussian_filter
from napari.layers import Image


class BlurSlider(QDoubleSlider):
    def __init__(self, viewer, image_layer=None, blur_func=None, parent=None):
        super().__init__(parent=parent)
        self.viewer = viewer
        self._smoothed_layer = None
        self.image_layer = image_layer
        self.setMaximum(20)
        self.valueChanged.connect(self.update_image)
        self.setMouseTracking(True)
        self.setOrientation(Qt.Horizontal)
        self.blur_func = blur_func if blur_func is not None else lambda img, val: gaussian_filter(img, val)
        self.children()[0].installEventFilter(self)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
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

    def eventFilter(self, obj: 'QObject', event: 'QEvent') -> bool:
        if event.type() == QEvent.MouseButtonPress:
            self.mousePressEvent(event)
        elif event.type() == QEvent.MouseButtonRelease:
            self.mouseReleaseEvent(event)
        return super().eventFilter(obj, event)
