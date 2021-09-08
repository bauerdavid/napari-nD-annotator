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
            call_button="Start",
            param_options={
                "channels_dim":{"max": viewer.dims.ndim-1}
            }
        )
        self.image.native.currentIndexChanged.connect(self.set_has_channel)

    def select_cells_widget(self, bounding_boxes: BoundingBoxLayer, image: Image, mask: Labels,
                            object_name="Object #", has_channels=True, channels_dim=0) -> Labels:
        print(bounding_boxes, image, mask)
        if None in [bounding_boxes, image, mask]:
            raise ValueError("Each one of 'bounding boxes', 'image' and 'mask' layers must be provided")
        image.visible = False
        mask.visible = False
        widget = ListWidget(self.viewer, object_name, bounding_boxes, image, mask, channels_dim if has_channels else None)
        dock_widget = self.viewer.window.add_dock_widget(widget)
        dock_widget.installEventFilter(widget)

    def set_has_channel(self, event):
        layer = self.image.value
        if not any(dim <= 3 for dim in layer.data.shape):
            self.has_channels.value = False
        else:
            self.has_channels.value = True
            self.channels_dim.value = list(filter(lambda x: x <= 3, layer.data.shape))[0]
        self.image.native.currentIndexChanged.disconnect(self.set_has_channel)

    def objectName(self):
        return "object crop widget"

