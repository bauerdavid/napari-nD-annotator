import math
import warnings

import numpy as np
import tifffile
from magicgui.widgets import FunctionGui
from napari.layers import Image, Labels

from bounding_boxes import BoundingBoxLayer
from widgets.object_list import ListWidget


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

    def select_cells_widget(self, bounding_boxes: BoundingBoxLayer, image: Image, mask: Labels,
                            object_name="Object #", channels_dim=0) -> Labels:
        if None in [bounding_boxes, image, mask]:
            raise ValueError("Each one of 'bounding boxes', 'image' and 'mask' layers must be provided")
        image.visible = False
        mask.visible = False
        widget = ListWidget(self.viewer, object_name, bounding_boxes, image, mask, channels_dim)
        dock_widget = self.viewer.window.add_dock_widget(widget)
        dock_widget.installEventFilter(widget)
