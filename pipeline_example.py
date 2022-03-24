import napari
import numpy as np

from napari.layers import Labels, Shapes

from widgets import ContourWidget, ObjectExtractorWidget, CreateBBoxesWidget, InterpolationWidget

try:
    import settings
except ModuleNotFoundError as e:
    print("create settings.py first")
    raise e

viewer = napari.Viewer()
image_layer = viewer.add_image(settings.test_image, colormap=settings.colormap, rgb=settings.rgb, name="Image")
viewer.window.add_dock_widget(CreateBBoxesWidget())
viewer.window.add_dock_widget(ObjectExtractorWidget(viewer))
viewer.window.add_dock_widget(InterpolationWidget(viewer))
viewer.window.add_dock_widget(ContourWidget(viewer))
labels_layer = Labels(data=np.zeros(image_layer.extent.data[1].astype(int) + 1, dtype=np.uint16))
viewer.add_layer(labels_layer)
viewer.add_layer(Shapes(ndim=3))
napari.run()
