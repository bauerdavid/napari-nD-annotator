from itertools import product

import numpy as np
from magicgui.widgets import FunctionGui
from napari.layers import Points, Image

from boundingbox.bounding_boxes import BoundingBoxLayer


class CreateBBoxesWidget(FunctionGui):
    def __init__(self):
        super().__init__(
            self.create_bboxes,
            call_button="Create bounding boxes"
        )
    def create_bboxes(self, points: Points, image: Image, size=50) -> BoundingBoxLayer:
        if points is None or image is None:
            return
        p_data = points.data.round().astype(int)
        im_size = np.asarray(image.data.shape)
        corner_idx = np.asarray(list(product([0, 1], repeat=points.ndim)))
        bboxes = np.asarray([np.where(corner_idx, np.maximum(p - size // 2, 0), np.minimum(p + size // 2, im_size)) for p in p_data])
        return BoundingBoxLayer(data=bboxes, name="BoundingBoxes", edge_color="lightgreen", face_color="transparent")

    def objectName(self):
        return "bounding box widget"
