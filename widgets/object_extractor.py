import math
import warnings

import numpy as np
import tifffile
from magicgui.widgets import FunctionGui
from napari.layers import Image, Labels

from boundingbox.bounding_boxes import BoundingBoxLayer
from .object_list import ListWidget
class ObjectExtractorWidget(FunctionGui):
    def __init__(self, viewer):
        self.viewer = viewer
        super().__init__(
            self.select_cells_widget,
            call_button="List objects"
        )
        self.image.native.currentIndexChanged.connect(self.on_layer_changed)
        self.has_channels.native.clicked.connect(self.on_has_channels_clicked)
        self.viewer.dims.events.ndisplay.connect(lambda _: self.call_button.native.setEnabled(self.viewer.dims.ndisplay == 2))

    def select_cells_widget(self, bounding_boxes: BoundingBoxLayer, image: Image, mask: Labels,
                            object_name="Object #", has_channels=True, channels_dim=0):
        if None in [bounding_boxes, image, mask]:
            raise ValueError("Each one of 'bounding boxes', 'image' and 'mask' layers must be provided")
        image.visible = False
        mask.visible = False
        widget = ListWidget(self.viewer, object_name, bounding_boxes, image, mask, channels_dim if has_channels else None)
        dock_widget = self.viewer.window.add_dock_widget(widget)
        dock_widget.installEventFilter(widget)

    def on_layer_changed(self, event):
        layer = self.image.value
        if layer.rgb:
            self.has_channels.value = True
            self.has_channels.native.setDisabled(True)
            self.channels_dim.native.setMinimum(-1)
            self.channels_dim.value = -1
            self.channels_dim.native.setDisabled(True)
        else:
            self.channels_dim.native.setMinimum(0)
            self.channels_dim.native.setMaximum(layer.data.ndim - 1)
            self.has_channels.native.setDisabled(False)
            if not any(dim <= 3 for dim in layer.data.shape):
                self.has_channels.value = False
            else:
                self.has_channels.value = True
                self.channels_dim.value = layer.data.shape.index(list(filter(lambda x: x <= 3, layer.data.shape))[0])
            self.channels_dim.native.setDisabled(not self.has_channels.value)
            # self.image.native.currentIndexChanged.disconnect(self.set_has_channel)

    def on_has_channels_clicked(self, state):
        self.channels_dim.native.setDisabled(not state)


    def objectName(self):
        return "object crop widget"

