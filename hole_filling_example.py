import napari
from napari.layers import Labels, Shapes
import numpy as np
from widgets.contour_widget import ContourWidget
try:
    import settings
except ModuleNotFoundError:
    print("create settings.py first")

viewer = napari.Viewer()
image_layer = viewer.add_image(settings.test_image, colormap=settings.colormap, rgb=settings.rgb, name="Image")
labels_layer = Labels(data=np.zeros(image_layer.extent.data[1].astype(int) + 1, dtype=np.uint16))
labels_layer.brush_size = 2
viewer.add_layer(labels_layer)
shapes_layer = Shapes(ndim=image_layer.ndim)
viewer.add_layer(shapes_layer)
viewer.window.add_dock_widget(ContourWidget(viewer))
napari.run()
