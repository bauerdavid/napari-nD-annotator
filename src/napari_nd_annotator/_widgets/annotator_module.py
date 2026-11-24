import json
import traceback
from pathlib import Path
from typing import Union, Optional

import napari
from magicclass import magicclass, MagicTemplate, vfield, set_design, field
from magicclass.serialize import serialize, deserialize
from magicgui._util import debounce
from napari.layers.labels._labels_utils import sphere_indices
from qtpy.QtCore import QObject, QEvent, Qt
from qtpy.QtWidgets import QSizePolicy
from napari.layers import Labels
import numpy as np
from skimage.draw import draw
import math
import warnings
from dataclasses import field as dfield

from .interpolation_widget import InterpolationWidget
from .minimal_contour_widget import MinimalContourWidget
from .minimal_surface_widget import MinimalSurfaceWidget
from .working_set_panel import WorkingSetPanel
from .selected_label_bbox import install_selected_label_bbox
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
)
from ._utils.help_dialog import HelpDialog
from ._utils.shortcuts import LABELS, bind_all, unbind_all
from .._helper_functions import layer_ndisplay, layer_dims_displayed, layer_slice_indices, layer_dims_order, \
    _coerce_indices_for_vectorization, layer_get_order, unwrap_viewer


_ANNOTATOR_LABELS_SHORTCUTS = ("extend_mask", "reduce_mask", "decrement_label", "increment_label", "previous_slice",
                               "next_slice", "increase_brush", "decrease_brush")


@magicclass(name="Annotation Toolbox")
class AnnotatorWidget(MagicTemplate):
    autofill_objects = vfield(bool).with_options(value=True, tooltip="When drawing labels,"
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
        self._viewer = unwrap_viewer(viewer)
        self.persist = persist
        install_selected_label_bbox(self._viewer)
        self._active_labels_layer = None

        self._viewer.layers.selection.events.changed.connect(self._on_layer_selection_change)
        self._viewer.layers.selection.events.connect(lock_layer)

        # self.native.installEventFilter(self) # TODO fix
        self._set_fill_objects(self.autofill_objects)
        self._on_layer_selection_change()

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
        # which layers the tools work on, and the menu to change them (the object list has the whole panel)
        self.working_set_panel = WorkingSetPanel(self._viewer, compact=True)
        self.native.layout().insertWidget(0, self.working_set_panel)
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
    def _dump(self, path: Optional[Union[str, Path]] = None) -> None:
        data = serialize(self)
        path = Path(path or self._dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as save_file:
            json.dump(data, save_file)

    def _load(self, path: Optional[Union[str, Path]] = None, quiet: bool = False) -> None:
        try:
            with open(path or self._dump_path) as save_file:
                data = json.load(save_file)
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

    def _unset_labels_callbacks(self):
        if self.active_labels_layer is None:
            return
        unbind_all(self.active_labels_layer, LABELS, _ANNOTATOR_LABELS_SHORTCUTS)

    def _set_labels_callbacks(self):
        if self.active_labels_layer is None:
            return
        bind_all(self.active_labels_layer, LABELS, {
            "extend_mask": extend_mask,
            "reduce_mask": reduce_mask,
            "decrement_label": decrement_selected_label,
            "increment_label": increment_selected_label,
            "previous_slice": scroll_to_prev(self._viewer),
            "next_slice": scroll_to_next(self._viewer),
            "increase_brush": increase_brush_size,
            "decrease_brush": decrease_brush_size,
        })

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
