import napari
import numpy as np

from napari.layers import Labels, Shapes
from skimage.data import cells3d

# cells_path = os.path.join(r"Y:\BIOMAG\Raman\fluorescent\No_20\B3\merged.tif")
from widgets.object_extractor import ObjectExtractorWidget
from widgets.bounding_box_creator import CreateBBoxesWidget

cells_path = r"20201022_Small_spheroid_T47D_111_Target_z00.tif"
size = 256
# map = astronaut()
image_map = cells3d()
'''

'''

viewer = napari.view_image(image_map, colormap='magma', rgb=False)
viewer.window.add_dock_widget(ObjectExtractorWidget(viewer))
viewer.window.add_dock_widget(CreateBBoxesWidget())

viewer.add_layer(Labels(data=np.zeros_like(image_map)))
viewer.add_layer(Shapes(ndim=3))
napari.run()
import ctypes
exit()


