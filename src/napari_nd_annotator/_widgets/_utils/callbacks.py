import warnings
from scipy.ndimage import binary_dilation, binary_erosion
from napari.layers import Labels, Image

LOCK_CHAR = u"\U0001F512"

def extend_mask(layer):
    if layer is None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = layer._slice.image.raw
    mask = labels == layer.selected_label
    mask = binary_dilation(mask)
    labels[mask] = layer.selected_label
    layer.events.data()
    layer.refresh()


def reduce_mask(layer):
    if layer is None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = layer._slice.image.raw
    mask = labels == layer.selected_label
    eroded_mask = binary_erosion(mask)
    labels[mask & ~eroded_mask] = 0
    layer.events.data()
    layer.refresh()


def increment_selected_label(layer):
    if layer is None:
        return
    layer.selected_label = layer.selected_label+1


def decrement_selected_label(layer):
    if layer is None:
        return
    layer.selected_label = max(0, layer.selected_label-1)


def scroll_to_prev(viewer):
    def scroll_to_prev(_):
        if len(viewer.dims.not_displayed) == 0:
            return
        viewer.dims.set_current_step(viewer.dims.not_displayed[0],
                                     viewer.dims.current_step[viewer.dims.not_displayed[0]]-1)
    return scroll_to_prev


def scroll_to_next(viewer):
    def scroll_to_next(_):
        if len(viewer.dims.not_displayed) == 0:
            return
        viewer.dims.set_current_step(viewer.dims.not_displayed[0],
                                     viewer.dims.current_step[viewer.dims.not_displayed[0]] + 1)
    return scroll_to_next


def increase_brush_size(layer):
    if layer is None:
        return
    diff = min(max(1, layer.brush_size // 10), 5)
    layer.brush_size = max(0, layer.brush_size + diff)


def decrease_brush_size(layer):
    if layer is None:
        return
    diff = min(max(1, layer.brush_size // 10), 5)
    layer.brush_size = max(0, layer.brush_size - diff)


def lock_layer(event):
    for layer in event.source:
        if layer.name.startswith(LOCK_CHAR):
            layer.editable = False


def keep_layer_on_top(layer):
    def on_top_callback(e):
        layer_list = e.source
        if layer not in layer_list:
            return
        with layer_list.events.moved.blocker(keep_layer_on_top):
            try:
                for i in reversed(range(len(layer_list))):
                    elem = layer_list[i]
                    if elem == layer:
                        break
                    if type(elem) not in [Labels, Image]:
                        continue
                    layer_index = layer_list.index(layer)
                    if i == layer_index+1:
                        layer_list.move(i, layer_index)
                    else:
                        layer_list.move(layer_index, i)
                    break
            except Exception as e:
                ...
    return on_top_callback
