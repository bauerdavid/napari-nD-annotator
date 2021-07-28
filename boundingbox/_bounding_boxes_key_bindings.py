from napari.layers.utils.layer_utils import register_layer_action
from napari.utils.translations import trans

from _bounding_box_constants import Mode

def register_bounding_boxes_action(layer_type, description):
    return register_layer_action(layer_type, description)

def register_bounding_boxes_actions(layer_type):

    @register_bounding_boxes_action(layer_type, trans._('Select bounding boxes'))
    def activate_bb_select_mode(layer):
        """Activate shape selection tool."""
        layer.mode = Mode.SELECT

    @register_bounding_boxes_action(layer_type, trans._('Select vertices'))
    def activate_bb_direct_mode(layer):
        """Activate shape selection tool."""
        layer.mode = Mode.DIRECT


    @register_bounding_boxes_action(layer_type, trans._('Pan/Zoom'))
    def activate_bb_pan_zoom_mode(layer):
        """Activate shape selection tool."""
        layer.mode = Mode.PAN_ZOOM


    @register_bounding_boxes_action(layer_type, trans._('Add bounding box'))
    def activate_add_bb_mode(layer):
        """Activate shape selection tool."""
        layer.mode = Mode.ADD_BOUNDING_BOX
