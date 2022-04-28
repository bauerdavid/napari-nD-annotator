import itertools

import napari
import numpy as np
from skimage.data import cells3d

from boundingbox.bounding_boxes import BoundingBoxLayer

viewer = napari.Viewer()
viewer.add_image(cells3d()[:, 1])
points = viewer.add_points(ndim=3)
bbs = BoundingBoxLayer(ndim=3, face_color="transparent", edge_color="green")
viewer.add_layer(bbs)
def pts_2_bb(p1, p2):
    center = (p1 + p2) / 2
    size = np.sqrt(((p1 - p2) ** 2).sum()) * 1.2
    bb = np.asarray(np.where(list(itertools.product((False, True), repeat=3)), center + size / 2, center - size / 2))
    bb = np.clip(bb, 0, viewer.layers["Image"].data.shape)
    return bb

def cb(event):
    if event.type == "data":
        if len(points.data) % 2 == 0 and len(bbs.data) < len(points.data)//2:
            p1 = points.data[-2]
            p2 = points.data[-1]

            bbs.add(pts_2_bb(p1, p2))
points.events.connect(cb)
napari.run()
