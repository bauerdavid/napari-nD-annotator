"""The working set: the image, the labels layer and the bounding box layer the annotation tools work on.

The layers are related, and the relation is stored where it belongs:

* a labels layer has one source image: ``labels.metadata["source_image_layer"]`` (the key the minimal contour tool
  has always used),
* a bounding box layer has one source image as well, ``boxes.metadata["source_image_layer"]`` (its corners are indices
  of that image),
* a labels layer knows the box layer of its objects: ``labels.metadata["object_boxes_layer"]``.

`working_set` finds the layers a tool should work on, in this order:

1. the object selected in the object list (its three layers),
2. the active layer of the viewer: a labels layer, a bounding box layer, or an image (with the labels layer that
   was used with it last, or the only one that has it as its source),
3. the working set that was in use last, while its layers are still in the viewer. So selecting a helper layer (a
   points layer, a distance map) does not make the tools forget what they work on.

Links to layers that were removed from the viewer are dropped (see `install`). `working_set_changed` is emitted
(with ``viewer=`` and ``working_set=``) when the result changes.
"""
import itertools
import weakref
from typing import Any, NamedTuple, Optional

import numpy as np
from napari.layers import Image, Labels, Layer
from napari.utils.events import EventEmitter
from napari.utils.notifications import show_info

from .._helper_functions import unwrap_viewer

try:
    from napari_bbox import BoundingBoxLayer
except ImportError:
    BoundingBoxLayer = None

SOURCE_IMAGE = "source_image_layer"
OBJECT_BOXES = "object_boxes_layer"
SCRATCH = "scratch_layer"

SELECTED_OBJECT = "selected object"
ACTIVE_LAYER = "active layer"
LAST_USED = "last used"

# Emitted (with ``viewer=`` and ``working_set=``) when the working set of a viewer changes.
working_set_changed = EventEmitter(source=None, type_name="working_set_changed")


class WorkingSet(NamedTuple):
    image: Optional[Layer] = None
    labels: Optional[Layer] = None
    boxes: Optional[Layer] = None
    # the selected object of the object list (a `SelectedObject`) when ``source`` is "selected object"
    object: Any = None
    source: str = ""

    @property
    def is_empty(self):
        return self.image is None and self.labels is None and self.boxes is None


_EMPTY = WorkingSet()
_state = weakref.WeakKeyDictionary()  # viewer -> {"last": WorkingSet, "emitted": key}
_installed = weakref.WeakSet()


# --- the links

def _linked(layer, key):
    return layer.metadata.get(key) if layer is not None else None


def linked_image(layer, viewer=None):
    """The source image of a labels or bounding box layer, or None (also if it is not in ``viewer``)."""
    image = _linked(layer, SOURCE_IMAGE)
    if image is None or (viewer is not None and image not in viewer.layers):
        return None
    return image


def linked_boxes(labels, viewer=None):
    """The bounding box layer of the objects of a labels layer, or None (also if it is not in ``viewer``)."""
    boxes = _linked(labels, OBJECT_BOXES)
    if boxes is None or (viewer is not None and boxes not in viewer.layers):
        return None
    return boxes


def link_layers(image=None, labels=None, boxes=None):
    """Records which layers belong together (the one place that writes the links). None leaves a link alone."""
    if image is not None:
        for layer in (labels, boxes):
            if layer is not None:
                layer.metadata[SOURCE_IMAGE] = image
    if labels is not None and boxes is not None:
        labels.metadata[OBJECT_BOXES] = boxes


class LinkError(ValueError):
    """Layers that cannot be linked (the message says why)."""


def incompatibility(image, labels):
    """Why ``image`` cannot be the source image of ``labels`` (a sentence), or None if it can."""
    if getattr(image, "multiscale", False):
        return "The image '%s' is multiscale." % image.name
    image_shape, labels_shape = tuple(image.data.shape[:image.ndim]), tuple(labels.data.shape)
    if image.ndim != labels.ndim or image_shape != labels_shape:
        return "The image '%s' %s and the labels layer '%s' %s have different shapes." % (
            image.name, image_shape, labels.name, labels_shape)
    return None


def align_scale(reference, layer):
    """Gives ``layer`` the scale of ``reference`` (the voxel indices of the two have to mean the same place)."""
    reference_scale = np.asarray(reference.scale, dtype=float)
    count = min(len(reference_scale), layer.ndim)
    new_scale = np.array(layer.scale, dtype=float)
    new_scale[-count:] = reference_scale[-count:]
    if np.allclose(new_scale, layer.scale):
        return
    old_scale = tuple(float(s) for s in layer.scale)
    layer.scale = new_scale
    show_info("The scale of layer '%s' was %s, set to %s like '%s'" % (
        layer.name, old_scale, tuple(float(s) for s in new_scale), reference.name))


def set_source_image(labels, image, viewer=None):
    """Makes ``image`` the source image of ``labels`` (and of its box layer); raises `LinkError` if they do not fit.

    The labels layer and its boxes take the scale of the image.
    """
    reason = incompatibility(image, labels)
    if reason is not None:
        raise LinkError(reason)
    boxes = linked_boxes(labels, viewer)
    link_layers(image=image, labels=labels, boxes=boxes)
    for layer in (labels, boxes):
        if layer is not None:
            align_scale(image, layer)


def set_boxes(labels, boxes, viewer=None):
    """Makes ``boxes`` the box layer of the objects of ``labels``; raises `LinkError` if they do not fit."""
    if boxes.ndim != labels.ndim:
        raise LinkError("The box layer '%s' has %d dimensions, the labels layer '%s' has %d."
                        % (boxes.name, boxes.ndim, labels.name, labels.ndim))
    image = linked_image(labels, viewer)
    link_layers(image=image, labels=labels, boxes=boxes)
    if image is not None:
        align_scale(image, boxes)


def unlink_boxes(labels):
    """The labels layer has no box layer any more (the box layer itself is left alone)."""
    labels.metadata.pop(OBJECT_BOXES, None)


def new_labels_for(viewer, image):
    """Adds an empty labels layer for ``image``: its shape, scale and translate, linked to it, and selects it."""
    if getattr(image, "multiscale", False):
        raise LinkError("The image '%s' is multiscale." % image.name)
    labels = viewer.add_labels(np.zeros(tuple(image.data.shape[:image.ndim]), dtype=np.uint16),
                               name="labels (%s)" % image.name, scale=image.scale, translate=image.translate)
    link_layers(image=image, labels=labels)
    viewer.layers.selection.select_only(labels)
    return labels


MAX_LABEL_FOR_BOXES = 10_000_000  # (``find_objects`` makes a list as long as the largest id)


def objects_in_labels(labels):
    """The objects painted in a labels layer: their ids and the corners of their boxes (voxel indices, inclusive).

    Returns ``([], [])`` if there are none, or if the ids are too large to look at them one by one.
    """
    from scipy.ndimage import find_objects
    data = labels.data[0] if getattr(labels, "multiscale", False) else labels.data
    try:
        data = np.asarray(data)
        if data.size == 0 or not np.issubdtype(data.dtype, np.integer) or int(data.max()) > MAX_LABEL_FOR_BOXES:
            return [], []
        found = find_objects(np.maximum(data, 0) if np.issubdtype(data.dtype, np.signedinteger) else data)
    except (ValueError, TypeError, MemoryError):
        return [], []
    ids, corners = [], []
    for idx, slices in enumerate(found, start=1):
        if slices is None:
            continue
        low, high = [s.start for s in slices], [s.stop - 1 for s in slices]
        corners.append(np.asarray(np.where(list(itertools.product((False, True), repeat=len(slices))), high, low),
                                  dtype=float))
        ids.append(idx)
    return ids, corners


def new_boxes_for(viewer, labels, from_objects=False):
    """Adds a box layer for the objects of ``labels``, in the transform of its image, and links it.

    With ``from_objects`` it has a box around each object that is painted in the labels layer already.
    """
    if BoundingBoxLayer is None:
        raise LinkError("Bounding boxes need the napari-bbox plugin.")
    image = linked_image(labels, viewer)
    if image is None:
        raise LinkError("The labels layer '%s' has no source image: the boxes take its scale." % labels.name)
    boxes = BoundingBoxLayer(ndim=labels.ndim, name="objects (%s)" % labels.name, edge_color="green",
                             face_color="transparent", scale=image.scale, translate=image.translate)
    if from_objects:
        ids, corners = objects_in_labels(labels)
        if ids:
            boxes.data = corners
            boxes.features["label"] = np.asarray(ids, dtype=int)
    viewer.add_layer(boxes)
    set_boxes(labels, boxes, viewer)
    return boxes


def auto_link(viewer, labels):
    """Gives a labels layer without a source image the only image that fits it (shape and scale), and says so.

    Returns the image, or None if there is no link and no single fitting image (the user has to choose).
    """
    image = linked_image(labels, viewer)
    if image is not None or labels.metadata.get(SOURCE_IMAGE) is not None:
        return image
    fitting = [layer for layer in viewer.layers if isinstance(layer, Image)
               and incompatibility(layer, labels) is None and np.allclose(layer.scale, labels.scale)]
    if len(fitting) != 1:
        return None
    set_source_image(labels, fitting[0], viewer)
    show_info("The source image of the labels layer '%s' is '%s' (the only image of the same shape and scale)"
              % (labels.name, fitting[0].name))
    return fitting[0]


def _prune_links(viewer, removed):
    for layer in viewer.layers:
        for key in (SOURCE_IMAGE, OBJECT_BOXES):
            if layer.metadata.get(key) is removed:
                del layer.metadata[key]


# --- the resolver

def _is_boxes(layer):
    return BoundingBoxLayer is not None and isinstance(layer, BoundingBoxLayer)


def _labels_of_image(viewer, image):
    return [layer for layer in viewer.layers if isinstance(layer, Labels) and _linked(layer, SOURCE_IMAGE) is image]


def _from_labels(viewer, labels):
    return WorkingSet(linked_image(labels, viewer), labels, linked_boxes(labels, viewer), None, ACTIVE_LAYER)


def _from_boxes(viewer, boxes):
    image = linked_image(boxes, viewer)
    candidates = [layer for layer in viewer.layers
                  if isinstance(layer, Labels) and _linked(layer, OBJECT_BOXES) is boxes]
    labels = candidates[0] if len(candidates) == 1 else None  # (two labels layers may share the boxes: not guessed)
    if image is None and labels is not None:
        image = linked_image(labels, viewer)
    return WorkingSet(image, labels, boxes, None, ACTIVE_LAYER)


def _from_image(viewer, image, last):
    candidates = _labels_of_image(viewer, image)
    if last is not None and last.image is image and last.labels in candidates:
        labels = last.labels  # the labels layer that was used with this image last
    else:
        labels = candidates[0] if len(candidates) == 1 else None  # more than one: ambiguous, none is guessed
    boxes = linked_boxes(labels, viewer) if labels is not None else None
    return WorkingSet(image, labels, boxes, None, ACTIVE_LAYER)


# A layer that a tool makes for a moment (the slice that is annotated, ...) is made with ``metadata={SCRATCH: True}``
# (not marked afterwards: the viewer reacts to a new layer being selected before the caller gets it back).
# A scratch layer is never part of a working set: selecting it neither changes what the object list shows nor
# deselects the selected object, and nothing is linked to it by guessing.


def is_scratch(layer):
    return layer is not None and bool(layer.metadata.get(SCRATCH))


def active_working_set(viewer):
    """The working set of the active layer if that is a labels layer, or a box layer that a labels layer refers to.

    None for any other active layer (and for a scratch layer): the selected object and the layers used last are not
    looked at (this is what the tools follow when the user selects a labels layer).
    """
    viewer = unwrap_viewer(viewer)
    active = viewer.layers.selection.active
    if is_scratch(active):
        return None
    if isinstance(active, Labels):
        return _from_labels(viewer, active)
    if _is_boxes(active):
        result = _from_boxes(viewer, active)
        return result if result.labels is not None else None
    return None


def is_helper_layer(layer):
    """True for the layers that are neither images nor labels (boxes, points, shapes...): selecting one of them is
    not leaving the data being annotated, so the tools keep working on the labels layer of the working set."""
    return layer is not None and not isinstance(layer, (Image, Labels))


def _still_valid(viewer, working_set):
    layers = [layer for layer in (working_set.image, working_set.labels, working_set.boxes) if layer is not None]
    return bool(layers) and all(layer in viewer.layers for layer in layers)


def working_set(viewer) -> WorkingSet:
    """The layers the tools of ``viewer`` work on (see the module docstring); empty if there are none."""
    from .object_list import selected_object_info  # (object_list uses the links of this module)
    viewer = unwrap_viewer(viewer)
    state = _state.setdefault(viewer, {})
    last = state.get("last")
    selected = selected_object_info(viewer)
    if selected is not None:
        result = WorkingSet(selected.image_layer, selected.labels_layer, selected.bbox_layer, selected,
                            SELECTED_OBJECT)
    else:
        active = viewer.layers.selection.active
        result = None
        if is_scratch(active):
            pass  # not a layer to work on: what was used last stays
        elif isinstance(active, Labels):
            result = _from_labels(viewer, active)
        elif _is_boxes(active):
            result = _from_boxes(viewer, active)
        elif isinstance(active, Image):
            result = _from_image(viewer, active, last)
        if result is None or result.is_empty:
            result = last._replace(source=LAST_USED, object=None) \
                if last is not None and _still_valid(viewer, last) else _EMPTY
    if not result.is_empty:
        state["last"] = result
    return result


# --- following the viewer

def _key(result):
    selected = result.object
    return (id(result.image), id(result.labels), id(result.boxes),
            None if selected is None else (int(selected.label), tuple(selected.bbox_min), tuple(selected.bbox_max)))


def _emit_if_changed(viewer):
    viewer = unwrap_viewer(viewer)
    result = working_set(viewer)
    state = _state.setdefault(viewer, {})
    key = _key(result)
    if state.get("emitted") == key:
        return
    state["emitted"] = key
    working_set_changed(viewer=viewer, working_set=result)


def notify(viewer):
    """Tells the listeners of `working_set_changed` (after the links were changed), if the working set changed."""
    _emit_if_changed(viewer)


def _on_selected_object_changed(event):
    viewer = getattr(event, "viewer", None)
    if viewer is not None:
        _emit_if_changed(viewer)


def install(viewer):
    """Makes ``viewer`` drop the links to removed layers and emit `working_set_changed` (once per viewer)."""
    viewer = unwrap_viewer(viewer)
    if viewer is None or viewer in _installed:
        return
    _installed.add(viewer)

    def removed(event):
        _prune_links(viewer, event.value)
        _emit_if_changed(viewer)

    viewer.layers.events.removed.connect(removed)
    viewer.layers.events.inserted.connect(lambda event: _emit_if_changed(viewer))
    viewer.layers.selection.events.active.connect(lambda event: _emit_if_changed(viewer))
    from .object_list import selected_object_changed
    selected_object_changed.connect(_on_selected_object_changed)  # (connecting twice does nothing)
