import napari
from boundingbox.bounding_boxes import BoundingBoxLayer

viewer = napari.Viewer()

viewer.add_layer(BoundingBoxLayer(ndim=4, edge_width=3, edge_color="lightgreen", opacity=1, face_color="transparent"))
napari.run()
