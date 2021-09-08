import napari
import numpy as np
from napari.layers import Image, Labels
from skimage.data import cells3d
from widgets.projections import SliceDisplayWidget
viewer = napari.Viewer()

image_map = cells3d()[:, 1, ...]
image_layer = Image(image_map, colormap='magma', rgb=False)
mask_layer = Labels(np.zeros_like(image_map))
viewer.add_layer(image_layer)
viewer.add_layer(mask_layer)
viewer.window.add_dock_widget(SliceDisplayWidget(viewer, image_layer, mask_layer))
napari.run()
