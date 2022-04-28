import napari
import numpy as np
from PyQt5.QtWidgets import QDockWidget

from napari.layers import Labels, Shapes

from object_list_bb import ListWidgetBB
from annotator_module import AnnotatorModule
try:
    import settings
except ModuleNotFoundError as e:
    print("create settings.py first")
    raise e
# w = QDockWidget()
# w.area
viewer = napari.Viewer()
viewer.window.add_dock_widget(AnnotatorModule(viewer))
image_layer = viewer.add_image(settings.test_image, channel_axis=settings.channels_dim, colormap=settings.colormap, rgb=settings.rgb, name="Image")
labels_layer = Labels(data=np.zeros(image_layer.extent.data[1].astype(int), dtype=np.uint16))
viewer.add_layer(labels_layer)
viewer.window.add_dock_widget(ListWidgetBB(viewer), area="left")
napari.run()
