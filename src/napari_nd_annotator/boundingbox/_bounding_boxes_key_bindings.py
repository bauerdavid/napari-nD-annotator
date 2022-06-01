# A copy of napari.layers.shapes._shapes_key_bindings
from napari.layers.utils.layer_utils import register_layer_action
from napari.utils.translations import trans

from ._bounding_box_constants import Mode

def register_bounding_boxes_action(layer_type, description):
    return register_layer_action(layer_type, description)
'''
def register_bounding_boxes_actions(layer_type):

    @register_bounding_boxes_action(layer_type, trans._('Select bounding boxes'))
    def activate_bb_select_mode(layer):
        """Activate bounding box selection tool."""
        layer.mode = Mode.SELECT

    @register_bounding_boxes_action(layer_type, trans._('Pan/Zoom'))
    def activate_bb_pan_zoom_mode(layer):
        """Activate pan and zoom mode."""
        layer.mode = Mode.PAN_ZOOM


    @register_bounding_boxes_action(layer_type, trans._('Add bounding box'))
    def activate_add_bb_mode(layer):
        """Activate add bounding box tool."""
        layer.mode = Mode.ADD_BOUNDING_BOX
'''
