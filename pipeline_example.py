import napari
import numpy as np

from napari.layers import Labels, Shapes
from skimage.data import cells3d

from widgets.contour_widget import ContourWidget
from widgets.object_extractor import ObjectExtractorWidget
from widgets.bounding_box_creator import CreateBBoxesWidget

image_map = cells3d()[:, 1, ...]

viewer = napari.view_image(image_map, colormap='magma', rgb=False)
viewer.window.add_dock_widget(ObjectExtractorWidget(viewer))
viewer.window.add_dock_widget(CreateBBoxesWidget())
viewer.window.add_dock_widget(ContourWidget(viewer))
labels_layer = Labels(data=np.zeros_like(image_map))
viewer.add_layer(labels_layer)
viewer.add_layer(Shapes(ndim=3))
napari.run()
