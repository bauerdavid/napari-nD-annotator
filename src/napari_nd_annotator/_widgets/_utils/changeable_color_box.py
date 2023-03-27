import warnings

import numpy as np
from napari.utils.events import disconnect_events

from napari.utils.translations import trans
from qtpy.QtWidgets import QWidget
from qtpy.QtCore import Qt
from qtpy.QtGui import QPainter, QColor


class QtChangeableColorBox(QWidget):
    """A widget that shows a square with the current label color.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.
    """

    def __init__(self, layer):
        super().__init__()

        self._layer = None
        self.layer = layer

        self.setAttribute(Qt.WA_DeleteOnClose)

        self._height = 24
        self.setFixedWidth(self._height)
        self.setFixedHeight(self._height)
        self.setToolTip(trans._('Selected label color'))

        self.color = None

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, new_layer):
        if new_layer == self.layer:
            return
        if self._layer is not None:
            self._layer.events.selected_label.disconnect(self._on_selected_label_change)
            self._layer.events.opacity.disconnect(self._on_opacity_change)
            self._layer.events.colormap.disconnect(self._on_colormap_change)
        self._layer = new_layer
        if new_layer is not None:
            new_layer.events.selected_label.connect(
                self._on_selected_label_change
            )
            new_layer.events.opacity.connect(self._on_opacity_change)
            new_layer.events.colormap.connect(self._on_colormap_change)

    def _on_selected_label_change(self, *args):
        """Receive layer model label selection change event & update colorbox."""
        self.update()

    def _on_opacity_change(self, *args):
        """Receive layer model label selection change event & update colorbox."""
        self.update()

    def _on_colormap_change(self, *args):
        """Receive label colormap change event & update colorbox."""
        self.update()

    def paintEvent(self, event):
        """Paint the colorbox.  If no color, display a checkerboard pattern.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter = QPainter(self)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected_color = self.layer._selected_color if self.layer else None
        if selected_color is None:
            self.color = None
            for i in range(self._height // 4):
                for j in range(self._height // 4):
                    if (i % 2 == 0 and j % 2 == 0) or (
                        i % 2 == 1 and j % 2 == 1
                    ):
                        painter.setPen(QColor(230, 230, 230))
                        painter.setBrush(QColor(230, 230, 230))
                    else:
                        painter.setPen(QColor(25, 25, 25))
                        painter.setBrush(QColor(25, 25, 25))
                    painter.drawRect(i * 4, j * 4, 5, 5)
        else:
            color = np.multiply(selected_color, self.layer.opacity)
            color = np.round(255 * color).astype(int)
            painter.setPen(QColor(*list(color)))
            painter.setBrush(QColor(*list(color)))
            painter.drawRect(0, 0, self._height, self._height)
            self.color = tuple(color)

    def deleteLater(self):
        disconnect_events(self.layer.events, self)
        super().deleteLater()

    def closeEvent(self, event):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.events, self)
        super().closeEvent(event)
