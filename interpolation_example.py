import napari
import numpy as np
from napari.layers import Labels
from widgets.interpolation_widget import InterpolationWidget
from widgets.contour_widget import ContourWidget

try:
    import settings
except ModuleNotFoundError:
    print("create settings.py first")
viewer = napari.Viewer()
image_layer = viewer.add_image(settings.test_image, colormap=settings.colormap, name="Image")
labels_layer = Labels(data=np.zeros(image_layer.extent.data[1].astype(int) + 1, dtype=np.uint16))
labels_layer.brush_size = 1
labels_layer.mode = "paint"
viewer.add_layer(labels_layer)
viewer.layers.selection.select_only(labels_layer)
viewer.window.add_dock_widget(InterpolationWidget(viewer))
viewer.window.add_dock_widget(ContourWidget(viewer))

napari.run()
