import warnings
from scipy.ndimage import binary_dilation, binary_erosion


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
