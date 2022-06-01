import napari
import numpy as np

from napari.layers import Labels, Shapes

from object_list_bb import ListWidgetBB
from annotator_module import AnnotatorModule
from skimage.data import cells3d

# w = QDockWidget()
# w.area
viewer = napari.Viewer()
viewer.window.add_dock_widget(AnnotatorModule(viewer))
image_layer = viewer.add_image(cells3d(), channel_axis=1, colormap="magma", name="Image")

extent = image_layer[0].extent.data[1].astype(int) if type(image_layer) is list else image_layer.extent.data[1].astype(int)

labels_layer = Labels(data=np.zeros(extent, dtype=np.uint16))
viewer.add_layer(labels_layer)
list_widget = ListWidgetBB(viewer)
viewer.window.add_dock_widget(list_widget, area="left")
napari.run()
