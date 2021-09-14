import napari
import numpy as np
from napari.layers import Image, Labels
from widgets.projections import SliceDisplayWidget

try:
    import settings
except ModuleNotFoundError:
    print("create settings.py first")

viewer = napari.Viewer()
image_layer = Image(settings.test_image, colormap=settings.colormap, rgb=settings.rgb, name="Image")
labels_layer = Labels(data=np.zeros(image_layer.extent.data[1].astype(int) + 1, dtype=np.uint16))
viewer.add_layer(image_layer)
viewer.add_layer(labels_layer)
viewer.window.add_dock_widget(SliceDisplayWidget(viewer, image_layer, labels_layer, channels_dim=settings.channels_dim))
napari.run()
