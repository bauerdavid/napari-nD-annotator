import json
import traceback
from pathlib import Path

import itertools

import napari
from magicclass import magicclass, MagicTemplate, vfield, set_design, field
from magicclass.serialize import serialize, deserialize
from magicgui._util import debounce
from napari.layers.labels._labels_utils import sphere_indices
from qtpy.QtCore import QObject, QEvent, Qt
from qtpy.QtWidgets import QSizePolicy
from napari.layers import Labels, Image
from napari.qt.threading import create_worker
import numpy as np
from skimage.draw import draw
from scipy.ndimage import binary_fill_holes, find_objects
import math
from packaging import version
import warnings
from dataclasses import field as dfield

from .._napari_version import NAPARI_VERSION

if NAPARI_VERSION >= version.parse("0.4.15"):
    try:
        from napari_bbox import BoundingBoxLayer
    except ImportError:
        BoundingBoxLayer = None
else:
    BoundingBoxLayer = None

from .interpolation_widget import InterpolationWidget
from .minimal_contour_widget import MinimalContourWidget
from .minimal_surface_widget import MinimalSurfaceWidget
from ._utils.callbacks import (
    extend_mask,
    reduce_mask,
    increase_brush_size,
    decrease_brush_size,
    scroll_to_next,
    scroll_to_prev,
    increment_selected_label,
    decrement_selected_label,
    lock_layer,
    LOCK_CHAR
)
from ._utils.help_dialog import HelpDialog
from .._helper_functions import layer_ndisplay, layer_dims_displayed, layer_slice_indices, layer_dims_order, \
    _coerce_indices_for_vectorization, layer_get_order


@magicclass(name="Annotation Toolbox")
class AnnotatorWidget(MagicTemplate):
    autofill_objects = vfield(bool).with_options(tooltip="When drawing labels,"
                                              " close the drawn curve and fill its area after releasing the mouse")

    @magicclass(widget_type="tabbed")
    class ToolsWidget:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interpolation_widget = field(InterpolationWidget, name="Interpolation")
            minimal_contour_widget = field(MinimalContourWidget, name="Minimal Contour")
            minimal_surface_widget = field(MinimalSurfaceWidget, name="Minimal Surface") if MinimalSurfaceWidget is not None else None
        persist = dfield(default=bool)

    def __init__(self, viewer: napari.Viewer, persist=True):
        self._viewer = viewer
        self.persist = persist
        self._active_labels_layer = None
        self._active_bbox_layer = None

        self._viewer.layers.selection.events.connect(self._on_layer_selection_change)
        self._viewer.layers.selection.events.connect(lock_layer)
        self._viewer.layers.events.inserted.connect(self._move_bbox_to_top)
        self._viewer.layers.events.moved.connect(self._move_bbox_to_top)

        # self.native.installEventFilter(self) # TODO fix
        self._set_fill_objects(self.autofill_objects)
        self._on_layer_selection_change()
        # self._init_bbox_layer()

    def __post_init__(self):
        if self.persist:
            try:
                self._load(quiet=True)
            except Exception as e:
                traceback.print_exc()
        self.changed.connect(self._on_change)
        for w in self._list:
            if w.name == "show_help_window":
                w.native.setFixedSize(20, 20)
                w.native.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                self.native.layout().setAlignment(w.native, Qt.AlignRight)
        self.ToolsWidget.interpolation_widget._initialize(self._viewer, self)
        self.ToolsWidget.minimal_contour_widget._initialize(self._viewer)
        if self.ToolsWidget.minimal_surface_widget is not None:
            self.ToolsWidget.minimal_surface_widget._initialize(self._viewer, self.ToolsWidget.minimal_contour_widget)

    @debounce(wait=0.5)
    def _on_change(self, *args, **kwargs) -> None:
        if self.persist:
            self._dump()

    @property
    def _dump_path(self) -> Path:
        from magicgui._util import user_cache_dir

        name = getattr(self.__class__, "__qualname__", str(self.__class__))
        name = name.replace("<", "-").replace(">", "-")  # e.g. <locals>
        return user_cache_dir("napari") / f"{self.__class__.__module__}.{name}"

    @debounce
    def _dump(self, path: str | Path | None = None) -> None:
        data = serialize(self)
        with open(path or self._dump_path, "w") as save_file:
            json.dump(data, save_file)

    def _load(self, path: str | Path | None = None, quiet: bool = False) -> None:
        try:
            with open(path or self._dump_path) as save_file:
                data = json.load(save_file)
                print(data)
                deserialize(self, data)
        except Exception as e:
            if not quiet:
                raise e

    @set_design(visible=False)
    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Hide:
            self._disconnect_all()
            return True
        elif event.type() == QEvent.Show:
            self._connect_all()
            return True
        return self.native.eventFilter(source, event)

    @property
    def active_labels_layer(self):
        return self._active_labels_layer

    @active_labels_layer.setter
    def active_labels_layer(self, labels_layer):
        if not isinstance(labels_layer, Labels) and labels_layer is not None:
            raise TypeError("'active_labels_layer' should be a Labels layer.")
        if labels_layer == self.active_labels_layer:
            return
        self._unset_labels_callbacks()
        self._active_labels_layer = labels_layer
        self._set_fill_objects(self.autofill_objects)
        self._set_labels_callbacks()
        # self._init_bbox_layer()

    def _unset_labels_callbacks(self):
        if self.active_labels_layer is None:
            return
        self.active_labels_layer.bind_key("Control-+", overwrite=True)(None)
        self.active_labels_layer.bind_key("Control--", overwrite=True)(None)
        self.active_labels_layer.bind_key("Q", overwrite=True)(None)
        self.active_labels_layer.bind_key("E", overwrite=True)(None)
        self.active_labels_layer.bind_key("A", overwrite=True)(None)
        self.active_labels_layer.bind_key("D", overwrite=True)(None)

    def _set_labels_callbacks(self):
        if self.active_labels_layer is None:
            return
        if self._start_update_bbox not in self.active_labels_layer.events.selected_label.callbacks:
            self.active_labels_layer.events.selected_label.connect(self._start_update_bbox)
            self.active_labels_layer.events.data.connect(self._start_update_bbox)
        self.active_labels_layer.bind_key("Control-+", overwrite=True)(extend_mask)
        self.active_labels_layer.bind_key("Control--", overwrite=True)(reduce_mask)
        self.active_labels_layer.bind_key("Q", overwrite=True)(decrement_selected_label)
        self.active_labels_layer.bind_key("E", overwrite=True)(increment_selected_label)
        self.active_labels_layer.bind_key("A", overwrite=True)(scroll_to_prev(self._viewer))
        self.active_labels_layer.bind_key("D", overwrite=True)(scroll_to_next(self._viewer))
        self.active_labels_layer.bind_key("W", overwrite=True)(increase_brush_size)
        self.active_labels_layer.bind_key("S", overwrite=True)(decrease_brush_size)

    def _move_bbox_to_top(self, e=None):
        if self._active_bbox_layer not in self._viewer.layers:
            return
        layer_list = self._viewer.layers
        with layer_list.events.moved.blocker(self._move_bbox_to_top):
            try:
                for i in reversed(range(len(layer_list))):
                    layer = layer_list[i]
                    if layer == self._active_bbox_layer:
                        break
                    if type(layer) not in [Labels, Image]:
                        continue
                    bbox_index = layer_list.index(self._active_bbox_layer)
                    if i == bbox_index+1:
                        layer_list.move(i, bbox_index)
                    else:
                        layer_list.move(bbox_index, i)
                    break
            except KeyError:
                ...

    def _start_update_bbox(self, event):
        if self._active_bbox_layer is None:
            return
        worker = create_worker(self._update_bbox, event.source)
        worker.finished.connect(self._active_bbox_layer.refresh)
        worker.start()

    def _update_bbox(self, layer):
        if self._active_bbox_layer is None:
            return
        if len(self._active_bbox_layer.data) > 0:
            self._active_bbox_layer.data = []
        label = layer.selected_label
        data = layer.data
        mask = data == label
        bb_corners = find_objects(mask, max_label=1)[0]
        if bb_corners is None:
            return
        min_ = [slice_.start for slice_ in bb_corners]
        max_ = [slice_.stop - 1 for slice_ in bb_corners]
        bb = np.asarray(np.where(list(itertools.product((False, True), repeat=layer.ndim)), max_, min_))
        self._active_bbox_layer.add(bb)
        self._active_bbox_layer.properties["label"][0] = label
        self._active_bbox_layer.refresh_text()

    @autofill_objects.connect
    def _set_fill_objects(self, state):
        if self.active_labels_layer is None:
            return
        if state and self.visible:
            if self._fill_holes not in self.active_labels_layer.mouse_drag_callbacks:
                self.active_labels_layer.mouse_drag_callbacks.append(self._fill_holes)
        elif not state:
            if self._fill_holes in self.active_labels_layer.mouse_drag_callbacks:
                self.active_labels_layer.mouse_drag_callbacks.remove(self._fill_holes)

    def _disconnect_all(self):
        if self.active_labels_layer is not None and self._fill_holes in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.remove(self._fill_holes)

    def _connect_all(self):
        if self.autofill_objects and self.active_labels_layer is not None and self._fill_holes not in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.append(self._fill_holes)

    def _on_layer_selection_change(self, event=None):
        if event is None or event.type == "changed":
            active_layer = self._viewer.layers.selection.active
            if isinstance(active_layer, Labels):
                self.active_labels_layer = active_layer
            else:
                self.active_labels_layer = None

    @staticmethod
    def _draw_line(x1, y1, x2, y2, brush_size, output):
        line_x, line_y = draw.line(x1, y1, x2, y2)
        for x, y in zip(line_x, line_y):
            cx, cy = draw.disk((x, y), np.floor(brush_size / 2) + 0.5, output.shape)
            output[cx, cy] = True

    def _fill_holes(self, layer: Labels, event):
        if layer.mode != "paint" or layer_ndisplay(layer) != 2:
            return
        coordinates = layer.world_to_data(event.position)
        poly = []
        # coordinates = tuple(max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
        dims_displayed = layer_dims_displayed(layer)
        image_coords = tuple(int(coordinates[i]) for i in range(layer.ndim) if i in dims_displayed)
        slice_dims = layer_slice_indices(layer)
        current_draw = np.zeros_like(layer.data[slice_dims], bool)
        start_x, start_y = prev_x, prev_y = image_coords
        poly.append(image_coords)
        yield
        while event.type == 'mouse_move':
            coordinates = layer.world_to_data(event.position)
            # coordinates = tuple(max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
            image_coords = tuple(int(coordinates[i]) for i in range(layer.ndim) if i in dims_displayed)
            poly.append(image_coords)
            prev_x, prev_y = image_coords
            yield
        # s = np.asarray([[0, 1, 0],
        #                 [1, 1, 1],
        #                 [0, 1, 0]])
        s = None
        coordinates = layer.world_to_data(event.position)
        # coordinates = tuple(
        #     max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
        image_coords = tuple(int(coordinates[i]) for i in range(layer.ndim) if i in dims_displayed)
        poly.append(image_coords)
        prev_x, prev_y = image_coords
        # AnnotatorWidget._draw_line(prev_x, prev_y, start_x, start_y, layer.brush_size, current_draw)
        radius = np.floor(layer.brush_size / 2) + 0.5
        lx, ly = draw.line(prev_x, prev_y, start_x, start_y)
        poly = np.asarray(poly)
        px, py = draw.polygon(poly[:, 0], poly[:, 1], current_draw.shape)
        current_draw[px, py] = True
        sphere_mask_idx = sphere_indices(radius, (1., 1.))
        for p in zip(lx, ly):
            mask_indices = sphere_mask_idx + np.round(p).astype(
                int
            )
            mask_indices = mask_indices[np.all(np.logical_and(mask_indices>=0, mask_indices<current_draw.shape), axis=1)]
            current_draw[mask_indices[:, 0], mask_indices[:, 1]] = True
        # binary_fill_holes(current_draw, output=current_draw, structure=s)
        if layer.preserve_labels:
            current_draw = current_draw & (layer.data[slice_dims] == 0)
        idx = np.nonzero(current_draw)
        change_idx = np.zeros((layer.ndim, len(idx[0])), dtype=int)
        order = layer_dims_order(layer)
        dims_displayed = layer_dims_displayed(layer)
        slice_indices = layer_slice_indices(layer)
        for i, d in enumerate(order):
            change_idx[d] = idx[layer_get_order(layer)[i - layer.ndim + 2]] if d in dims_displayed else slice_indices[d]
        change_idx = _coerce_indices_for_vectorization(layer.data, change_idx)
        if hasattr(layer, "data_setitem"):
            layer.data_setitem(change_idx, layer.selected_label)
        else:
            layer._save_history(
                (change_idx, layer.data[change_idx], layer.selected_label))
            layer.data[change_idx] = layer.selected_label
            layer.events.data()
            layer.refresh()

    @set_design(text="?", width=20, height=20)
    def show_help_window(self):
        """Show keyboard shortcuts."""
        dialog = HelpDialog(self.native)
        dialog.show()

    def _init_bbox_layer(self):
        if self.active_labels_layer is None or BoundingBoxLayer is None:
            return
        if self._active_bbox_layer in self.viewer.layers:
            if (self.active_labels_layer is not None and self._active_bbox_layer.ndim == self.active_labels_layer.ndim
                    or self.active_labels_layer is None and self._active_bbox_layer.ndim == 2):
                return
            with self.viewer.layers.selection.events.blocker(self._on_layer_selection_change):
                self.viewer.layers.remove(self._active_bbox_layer)
        self._active_bbox_layer = BoundingBoxLayer(name="%s active object" % LOCK_CHAR, face_color="transparent",
                                                  edge_color="green", ndim=self.active_labels_layer.ndim if self.active_labels_layer else 2)
        self._active_bbox_layer.opacity = 1.
        self._active_bbox_layer.current_properties |= {"label": 0}
        self._active_bbox_layer.text = {
            "text": "{label:d}",
            "size": 10,
            "color": "green"
        }
        with self.viewer.layers.selection.events.blocker(self._on_layer_selection_change):
            prev_selection = self.viewer.layers.selection.copy()
            self.viewer.add_layer(self._active_bbox_layer)
            if len(prev_selection) == 1 and self._active_bbox_layer not in prev_selection:
                self.viewer.layers.selection = prev_selection

