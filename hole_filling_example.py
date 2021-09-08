import napari
import numpy as np

from napari.layers import Labels, Shapes
from skimage.data import cells3d
from widgets.contour_widget import ContourWidget

image_map = cells3d()
viewer = napari.view_image(image_map, colormap='magma', rgb=False)
labels_layer = Labels(data=np.zeros_like(image_map))
labels_layer.brush_size = 2
viewer.add_layer(labels_layer)
shapes_layer = Shapes(ndim=image_map.ndim)
viewer.add_layer(shapes_layer)
viewer.window.add_dock_widget(ContourWidget(viewer))
napari.run()
