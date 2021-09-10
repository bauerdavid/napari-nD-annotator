import napari
import numpy as np
from napari.layers import Labels
from skimage.data import cells3d
from widgets.interpolation_widget import InterpolationWidget
from widgets.contour_widget import ContourWidget
image_map = cells3d()[:, 1, ...]
viewer = napari.view_image(image_map, colormap='magma', rgb=False)
labels_layer = Labels(data=np.zeros_like(image_map))
labels_layer.brush_size = 1
labels_layer.mode = "paint"
viewer.add_layer(labels_layer)
viewer.layers.selection.select_only(labels_layer)
viewer.window.add_dock_widget(InterpolationWidget(viewer))
viewer.window.add_dock_widget(ContourWidget(viewer))

napari.run()
