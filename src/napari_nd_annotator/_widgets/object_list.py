import os.path
import warnings

import cv2
import napari
import numpy as np

from qtpy.QtCore import QEvent, Qt, QObject
from qtpy.QtGui import QImage, QPixmap, QHideEvent, QShowEvent
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QListWidgetItem, QListWidget, QMenu, \
    QComboBox, QPushButton, QStackedWidget, QDockWidget, QFileDialog, QSpinBox, QCheckBox
from napari import Viewer
from napari.layers import Image, Labels
from napari.layers.labels._labels_constants import Mode
from scipy.ndimage import find_objects

from ._utils import WidgetWithLayerList
from ._utils.clipping import LayerClipper, displayed_layer_dims
from ._utils.shortcuts import VIEWER, bind_all, unbind_all
from .._helper_functions import unwrap_viewer, viewer_qt_viewer, viewer_camera
from .projections import SliceDisplayWidget

try:
    from napari_bbox import BoundingBoxLayer
except ImportError:
    BoundingBoxLayer = None

import itertools
import csv
import weakref

import PIL.Image as PilImage

import time


_OBJECT_LISTS = weakref.WeakSet()


def selected_object(viewer):
    """The object selected in the object list of ``viewer`` as ``(label, bbox_min, bbox_max, image_layer)``, or None.

    The bounding box corners are inclusive and in data (index) coordinates of ``image_layer``, the image layer
    of the object list. Lets other widgets work on the selected object without a reference to the object list.
    """
    viewer = unwrap_viewer(viewer)
    for object_list in list(_OBJECT_LISTS):
        try:
            if object_list.viewer is not viewer or object_list.selected_idx is None:
                continue
            item = object_list.item(object_list.selected_idx)
            image_layer = object_list.image_layer
        except RuntimeError:  # the widget was deleted
            continue
        if item is None or image_layer is None:
            continue
        bounding_box = np.asarray(item.bounding_box)
        return item.idx, bounding_box[0].copy(), bounding_box[1].copy(), image_layer
    return None


def slice_len(slice_, count=None):
    if count is None:
        count = slice_.stop
    return len(range(*slice_.indices(count)))


if BoundingBoxLayer:
    class QObjectWidget(QWidget):
        def __init__(self, icon=None, name="Object", index=0, bounding_box=None, parent=None):
            super().__init__()
            self.textQVBoxLayout = QVBoxLayout()
            self.textUpQLabel = QLineEdit()
            self.textUpQLabel.setToolTip("Object name")
            self.id_label = QLabel()
            self.id_label.setText("Id: " + str(index))
            self.id_label.setToolTip("label index")
            self.textDownQLabel = QLabel()
            self.textQVBoxLayout.addWidget(self.textUpQLabel)
            self.textQVBoxLayout.addWidget(self.id_label)
            self.allQHBoxLayout  = QHBoxLayout()
            self.iconQLabel      = QLabel()
            self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
            self.allQHBoxLayout.addLayout(self.textQVBoxLayout, 1)
            self.setLayout(self.allQHBoxLayout)
            if icon is not None:
                self.iconQLabel.setPixmap(icon)
            self.setTextUp(name)
            if bounding_box is not None:
                self.setTextDown(bounding_box)
            self.name = name
            self.parent = parent

        @property
        def name(self):
            return self.textUpQLabel.text()

        @name.setter
        def name(self, name):
            self.textUpQLabel.setText(name)

        def setTextUp(self, text):
            self.textUpQLabel.setText(text)

        def setTextDown(self, bb):
            self.textDownQLabel.setText(("corner 1: %s\ncorner 2: %s" % tuple(bb)))

        def setIcon(self, icon):
            self.iconQLabel.setPixmap(icon)
            self.iconQLabel.setFixedSize(icon.size())


    class QObjectListWidgetItem(QListWidgetItem):
        def __init__(self, name, bounding_box, index, parent, viewer, image_layer, mask_layer, channels_dim, *args):
            super().__init__(None, *args)
            if name is None:
                name = parent.name_template
            self.object_item = QObjectWidget(name=name, index=index, bounding_box=bounding_box, parent=self)
            self.setSizeHint(self.object_item.sizeHint())
            self.setFlags(self.flags() | Qt.ItemFlag.ItemIsEditable)
            self.parent = parent
            self.viewer = viewer
            self.image_layer = image_layer
            self.mask_layer = mask_layer
            self.bounding_box = bounding_box
            self.channels_dim = channels_dim

        def __hash__(self):
            return id(self)

        @property
        def idx(self):
            return int(float(self.object_item.id_label.text().split(":")[-1]))

        @idx.setter
        def idx(self, idx):
            self.object_item.id_label.setText("Id: " + str(idx))

        @property
        def icon(self):
            return self._icon

        @icon.setter
        def icon(self, icon):
            if len(icon) == 0:
                return
            self._icon = icon.reshape(icon.shape[0], icon.shape[1], -1)
            if self._icon.shape[-1] == 1:
                self._icon = np.tile(self._icon, (1, 1, 3))
            elif self._icon.shape[-1] == 4:
                self._icon = self._icon[..., :-1]
            scale = 64 / max(self._icon.shape[1], self._icon.shape[0])
            out_size = (int(self._icon.shape[1]*scale), int(self._icon.shape[0]*scale))
            self._resized_icon = cv2.resize(self._icon, out_size, interpolation=cv2.INTER_NEAREST_EXACT)
            img = QImage(self._resized_icon, self._resized_icon.shape[1], self._resized_icon.shape[0], self._resized_icon.shape[1] * 3, QImage.Format.Format_RGB888)
            self.pixmap = QPixmap()
            self.pixmap = self.pixmap.fromImage(img, Qt.ImageConversionFlag.ColorOnly)
            self.object_item.setIcon(self.pixmap)
            # self.update_icon()

        @property
        def bounding_box(self):
            return self._bounding_box

        @bounding_box.setter
        def bounding_box(self, bounding_box):
            self._bounding_box = bounding_box
            self.object_item.setTextDown(bounding_box)
            self.object_item.update()
            if self.parent.crop_image_layer is not None and \
                self.parent.selected_idx == self.parent.indexFromItem(self).row():
                self.parent.crop_image_layer.data = self.image_layer.data[self.bbox_idx]
                self.parent.crop_image_layer.translate = self.crop_translate(self.image_layer)
                self.parent.crop_image_layer.refresh()
                self.parent.crop_mask_layer.data = self.mask_layer.data[self.bbox_idx]
                self.parent.crop_mask_layer.translate = self.crop_translate(self.mask_layer)
                self.parent.crop_mask_layer.refresh()
            if self.parent.clipper is not None and self.parent.selected_idx == self.parent.indexFromItem(self).row():
                self.update_clipping()
            self.update_icon()

        @property
        def bbox_idx(self):
            bbox_idx = tuple(
                slice(self.bounding_box[0, d], self.bounding_box[1, d]+1) for d in range(self.bounding_box.shape[1]))
            return bbox_idx

        @property
        def name(self):
            return self.object_item.name

        @name.setter
        def name(self, name):
            self.object_item.name = name

        def crop_translate(self, layer):
            # the bounding box is in data (index) coordinates, translate is in world coordinates
            return self.bounding_box[0, :]*layer.scale + layer.translate

        def create_layers(self):
            if self.image_layer is not None:
                image_layer = Image(self.image_layer.data[self.bbox_idx],
                                    name="Cropped %s %d" % (self.image_layer.name, self.idx),
                                    colormap=self.image_layer.colormap, rgb=self.image_layer.rgb,
                                    scale=self.image_layer.scale)
                image_layer.translate = self.crop_translate(self.image_layer)
                self.viewer.add_layer(image_layer)
            else:
                image_layer = None
            if self.mask_layer is not None:
                mask_layer = Labels(self.mask_layer.data[self.bbox_idx],
                                    name="Cropped %s %d" % (self.mask_layer.name, self.idx),
                                    scale=self.mask_layer.scale)
                mask_layer.selected_label = self.idx
                mask_layer.translate = self.crop_translate(self.mask_layer)
                mask_layer.brush_size = self.mask_layer.brush_size
                if image_layer is not None:
                    # lets the minimal contour widget find the image to work on
                    mask_layer.metadata["source_image_layer"] = image_layer
                self.mask_layer.events.data.connect(mask_layer.refresh)
                self.viewer.add_layer(mask_layer)
                # set after adding, so that the layer controls follow the mode
                try:
                    mask_layer.mode = self.mask_layer.mode
                except (ValueError, TypeError):
                    mask_layer.mode = Mode.PAINT
            else:
                mask_layer = None
            return image_layer, mask_layer

        def unselect(self):
            self.parent.release_clipper()
            had_crops = self.parent.crop_image_layer is not None
            if self.parent.crop_image_layer is not None:
                if self.parent.crop_image_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.parent.crop_image_layer)
                self.parent.crop_image_layer = None
            if self.parent.crop_mask_layer is not None:
                if self.parent.crop_mask_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.parent.crop_mask_layer)
                if self.mask_layer is not None:
                    self.mask_layer.events.data.disconnect(self.parent.crop_mask_layer.refresh)
                self.parent.crop_mask_layer.mouse_drag_callbacks.remove(QObjectListWidgetItem.update_layer)
                self.parent.crop_mask_layer = None
            if self.parent.projections_widget is not None:
                self.viewer.window.remove_dock_widget(self.parent.projections_widget)
                self.parent.projections_widget = None
            # napari stores the mode on hiding a layer and restores it on showing it, but setting
            # `visible = True` on a layer that is already visible resets the mode to pan/zoom
            if had_crops and self.mask_layer is not None and not self.mask_layer.visible:
                self.mask_layer.visible = True
            if had_crops and self.image_layer is not None and not self.image_layer.visible:
                self.image_layer.visible = True

        def on_select(self):
            self.unselect()
            if self.parent.use_clipping():
                self.select_with_clipping()
            else:
                self.select_with_crops()

        def update_clipping(self):
            self.parent.clipper.set_box(self.bounding_box[0], self.bounding_box[1])

        def select_with_clipping(self):
            # 3D: the original layers are clipped to the bounding box, no cropped copies are created
            clipper = LayerClipper(self.viewer)
            self.parent.clipper = clipper
            clipper.clip([self.image_layer, self.mask_layer], self.bounding_box[0], self.bounding_box[1])
            self.parent.update_ghosts()
            center = self.image_layer.data_to_world(self.bounding_box.mean(axis=0))
            self.viewer.dims.set_point(np.arange(self.image_layer.ndim, dtype=int), center)
            try:
                viewer_camera(self.viewer).center = tuple(center[d] for d in displayed_layer_dims(self.viewer, self.image_layer))
            except (ValueError, TypeError, IndexError):
                pass
            self.viewer.layers.selection.select_only(self.mask_layer)
            self.mask_layer.selected_label = self.idx

        def select_with_crops(self):
            self.parent.crop_image_layer, self.parent.crop_mask_layer = self.create_layers()
            self.parent.crop_image_layer.name = self.parent.crop_image_layer.name
            self.parent.crop_mask_layer.name = self.parent.crop_mask_layer.name
            self.viewer.dims.set_point(
                np.arange(self.parent.crop_image_layer.ndim, dtype=int),
                np.mean(self.parent.crop_image_layer.extent.world, axis=0))
            self.parent.crop_mask_layer.events.set_data.connect(self.on_data_change)
            if self.image_layer.ndim > 2:
                self.parent.projections_widget = SliceDisplayWidget(self.viewer, self.parent.crop_image_layer, self.parent.crop_mask_layer, self.channels_dim)
                self.viewer.window.add_dock_widget(self.parent.projections_widget)
            self.parent.crop_mask_layer.mouse_drag_callbacks.append(self.update_layer)
            self.viewer.layers.selection.select_only(self.parent.crop_mask_layer)
            # the minimal contour tool is not a napari mode (it borrows pan/zoom), so it is carried over separately
            mc_overlay = self.mask_layer._overlays.get("minimal_contour")
            if mc_overlay is not None and mc_overlay.enabled:
                mc_button = getattr(viewer_qt_viewer(self.viewer).controls.widgets.get(self.parent.crop_mask_layer),
                                    "mc_button", None)
                if mc_button is not None:
                    mc_button.click()
            self.mask_layer.selected_label = self.idx
            self.mask_layer.visible = False
            self.image_layer.visible = False

        def on_data_change(self, event):
            if self.parent.crop_mask_layer is None:
                return
            self.mask_layer.data[self.bbox_idx] = self.parent.crop_mask_layer.data
            if self.parent.projections_widget is not None:
                self.parent.projections_widget.update_slider_ranges()
            self.parent.on_layer_event(event)

        @staticmethod
        def update_layer(layer, event):
            yield
            while event.type == "mouse_move":
                yield
            layer.refresh()

        def update_icon(self):
            visible_dims = list(self.viewer.dims.displayed)[-2:]
            bbox_idx = tuple(
                slice(max(self.bounding_box[0, d], 0), self.bounding_box[1, d]+1) if d in visible_dims
                else self.bounding_box[:, d].mean().astype(int)
                for d in range(self.bounding_box.shape[1])
            )
            if self.image_layer is None:
                icon = np.zeros([slice_len(bbox_idx[d]) for d in range(len(bbox_idx)) if type(bbox_idx[d]) is slice] + [3])
            else:
                icon = self.image_layer.data[bbox_idx]
            if self.image_layer is not None and not self.image_layer.rgb:
                max_ = self.image_layer.contrast_limits[1]
                min_ = self.image_layer.contrast_limits[0]
                icon = np.clip((icon.astype(float)-min_)/(max_ - min_), 0, 1)
                icon = (self.image_layer.colormap.map(icon.ravel()).reshape(icon.shape + (4,)) * 255).astype(np.uint8)[..., :3]
            else:
                icon = icon.astype(np.uint8)[..., :3]
            if self.mask_layer is not None:
                overlay = self.mask_layer.data[bbox_idx] == self.idx
                img = PilImage.fromarray(icon).convert(mode="RGBA")
                mask = PilImage.fromarray(overlay).convert(mode="RGBA")
                mask.putalpha(PilImage.fromarray((overlay*127).astype(np.uint8)))
                img.alpha_composite(mask)
                icon = np.asarray(img)
            self.icon = icon


    class ObjectListWidget(QListWidget):
        def __init__(self, viewer, name_template, bounding_box_layer, image_layer, mask_layer, channels_dim, indices=None,
                     show_image_surroundings=True, show_labels_surroundings=False):
            super().__init__()
            # 3D: the original layers are clipped to the selected object (see `use_clipping`)
            self.clipper = None
            self._show_image_surroundings = show_image_surroundings
            self._show_labels_surroundings = show_labels_surroundings
            self._bounding_box_layer = None
            self._image_layer = None
            self._mask_layer = None
            self.crop_image_layer = None
            self.crop_mask_layer = None
            self.projections_widget = None
            self._mouse_down = False
            self.selected_idx = None
            _OBJECT_LISTS.add(self)
            self.previous_edge_color = None
            self._indices = None
            self.indices = indices
            self.viewer = viewer
            self.name_template = name_template
            self.channels_dim = channels_dim
            self.image_layer = image_layer
            self.mask_layer = mask_layer

            self.bounding_box_layer = bounding_box_layer
            self.setObjectName("Objects")
            self.update_items()
            self.itemClicked.connect(self.select_item)
            self.installEventFilter(self)
            self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)
            bind_all(self.viewer, VIEWER, {"toggle_bounding_boxes": self.toggle_bb_visibility})
            # bounding_box_layer.events.set_data.connect(on_data_change)

        @property
        def bounding_box_layer(self):
            return self._bounding_box_layer

        @bounding_box_layer.setter
        def bounding_box_layer(self, new_layer):
            if self._bounding_box_layer == new_layer:
                self.update_items()
                return
            if self._bounding_box_layer is not None:
                self._bounding_box_layer.events.set_data.disconnect(self._on_bounding_boxes_removed)
            self._bounding_box_layer = new_layer
            if new_layer is not None:
                new_layer.events.set_data.connect(self._on_bounding_boxes_removed)
                if self.bounding_box_change not in self._bounding_box_layer.mouse_drag_callbacks:
                    self._bounding_box_layer.mouse_drag_callbacks.append(self.bounding_box_change)
                if self._on_bb_double_click not in self._bounding_box_layer.mouse_double_click_callbacks:
                    self._bounding_box_layer.mouse_double_click_callbacks.append(self._on_bb_double_click)
                if "label" not in self._bounding_box_layer.features:
                    self._bounding_box_layer.features = {"label": np.arange(len(new_layer.data))}
                self._bounding_box_layer.current_properties |= {"label": 0}
                self._bounding_box_layer.refresh_text()
                if len(self._bounding_box_layer.data) > 0:
                    self._bounding_box_layer.text = {
                        "text": "{label:d}",
                        "size": 10,
                        "color": "green"
                    }
            self.update_items()

        @property
        def image_layer(self):
            return self._image_layer

        @image_layer.setter
        def image_layer(self, new_layer):
            clipped = self.clipper is not None and new_layer is not self._image_layer
            if clipped:
                self.release_clipper()
            self._image_layer = new_layer
            for i in range(self.count()):
                item = self.item(i)
                item.image_layer = new_layer
                item.update_icon()
            self.update_items()
            if clipped:
                self._reselect()

        @property
        def mask_layer(self):
            return self._mask_layer

        @mask_layer.setter
        def mask_layer(self, new_layer):
            clipped = self.clipper is not None and new_layer is not self._mask_layer
            if clipped:
                self.release_clipper()
            self._mask_layer = new_layer
            for i in range(self.count()):
                item = self.item(i)
                item.mask_layer = new_layer
                item.update_icon()
            self.update_items()
            if clipped:
                self._reselect()

        @property
        def show_image_surroundings(self):
            return self._show_image_surroundings

        @show_image_surroundings.setter
        def show_image_surroundings(self, show):
            self._show_image_surroundings = bool(show)
            self.update_ghosts()

        @property
        def show_labels_surroundings(self):
            return self._show_labels_surroundings

        @show_labels_surroundings.setter
        def show_labels_surroundings(self, show):
            self._show_labels_surroundings = bool(show)
            self.update_ghosts()

        def use_clipping(self):
            # clipping planes act only in the 3D canvas, in 2D the object is shown on cropped copies of the layers
            return self.viewer.dims.ndisplay == 3 and self.image_layer is not None and self.image_layer.ndim >= 3

        def update_ghosts(self):
            if self.clipper is None:
                return
            self.clipper.set_ghost(self.image_layer, self._show_image_surroundings)
            self.clipper.set_ghost(self.mask_layer, self._show_labels_surroundings)

        def release_clipper(self):
            if self.clipper is not None:
                clipper, self.clipper = self.clipper, None
                clipper.close()

        def _reselect(self):
            item = self.item(self.selected_idx) if self.selected_idx is not None else None
            if item is not None and self.image_layer is not None and self.mask_layer is not None:
                item.on_select()

        def _on_ndisplay_changed(self, event=None):
            # the selected object is shown differently in 2D and 3D
            try:
                self._reselect()
            except RuntimeError:  # the widget was deleted
                self.viewer.dims.events.ndisplay.disconnect(self._on_ndisplay_changed)

        @property
        def indices(self):
            return self._indices

        @indices.setter
        def indices(self, indices):
            if indices is None:
                indices = 1
            if type(indices) is int:
                starting_value = indices
                def index():
                    idx = starting_value
                    while True:
                        yield idx
                        idx += 1
                indices = index()
            self._indices = itertools.cycle(iter(indices))

        def on_layer_event(self, event):
            if event.type == "set_data" and not self._mouse_down:
                self.update_items(True)

        def select_item(self, item):
            selected_idx = self.indexFromItem(item).row()
            if self.mask_layer is not None and self.image_layer is not None:
                item.on_select()
            else:
                self.clearSelection()
                self.clearFocus()
                warnings.warn("Image and labels layer should be selected!")
                return
            if selected_idx == self.selected_idx or self.mask_layer is None or self.image_layer is None:
                self.selected_idx = None
                item.unselect()
                self.clearSelection()
                self.clearFocus()
                self.bounding_box_layer.edge_color[selected_idx] = self.previous_edge_color
                self.bounding_box_layer.data = self.bounding_box_layer.data
            else:
                if self.selected_idx is not None:
                    self.bounding_box_layer.edge_color[self.selected_idx] = self.previous_edge_color
                self.selected_idx = selected_idx
                self.previous_edge_color = self.bounding_box_layer.edge_color[selected_idx].copy()
                self.bounding_box_layer.edge_color[selected_idx] = (1., 0., 0., 1.)
                self.bounding_box_layer.data = self.bounding_box_layer.data

        def _on_bounding_boxes_removed(self, event=None):
            # bounding boxes deleted from the layer (e.g. with its delete button)
            if self._mouse_down or len(self._bounding_box_layer.data) >= self.count():
                return
            selected_item = self.item(self.selected_idx) if self.selected_idx is not None else None
            self.update_items()
            if selected_item is None:
                return
            try:
                row = self.row(selected_item)
            except RuntimeError:  # the item was deleted along with the whole list
                row = -1
            if row < 0:
                selected_item.unselect()
                self.selected_idx = None
            else:
                self.selected_idx = row

        def bounding_box_corners(self):
            if self._bounding_box_layer is None or len(self._bounding_box_layer.data) == 0:
                return []
            b_data = np.asarray(self._bounding_box_layer.data)
            if self.image_layer is not None:
                # the bounding boxes are in world coordinates, the item bounding boxes in data (index) ones
                b_data = (b_data - self.image_layer.translate) / self.image_layer.scale
            b_data = np.round(b_data).astype(int)
            return np.concatenate([b_data.min(1, keepdims=True), b_data.max(1, keepdims=True)], axis=1)

        def eventFilter(self, source: QObject, event: QEvent) -> bool:
            if event.type() == QEvent.Close:
                self.release_clipper()
                if self.crop_image_layer is not None:
                    self.viewer.layers.remove(self.crop_image_layer)
                    self.viewer.layers.remove(self.crop_mask_layer)
                    self.viewer.window.remove_dock_widget(self.projections_widget)
            elif event.type() == QEvent.ContextMenu and source is self:
                menu = QMenu()
                menu.addAction("Create layers")
                if menu.exec_(event.globalPos()):
                    item = self.itemAt(event.pos())
                    item.create_layers()
                return True
            elif event.type() == QEvent.Enter and type(source) == QObjectWidget:
                item = source.parent
                if item is None:
                    return False
                idx = self.indexFromItem(item).row()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.bounding_box_layer._value = (idx, None)
                    self.bounding_box_layer._set_highlight()
                return True
            elif event.type() == QEvent.Leave and type(source) == QObjectWidget:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.bounding_box_layer._value = (None, None)
                    self.bounding_box_layer._set_highlight()
                return True
            return super().eventFilter(source, event)

        def hideEvent(self, a0: QHideEvent) -> None:
            unbind_all(self.viewer, VIEWER, ["toggle_bounding_boxes"])
            super().hideEvent(a0)

        def toggle_bb_visibility(self, _=None):
            if self.bounding_box_layer is not None and self.bounding_box_layer.visible:
                mode = self.bounding_box_layer.mode
                self.bounding_box_layer.visible = False
                yield
                self.bounding_box_layer.visible = True
                self.bounding_box_layer.mode = mode

        def bounding_box_change(self, layer, event):
            previous_data = np.asarray([bb.copy() for bb in layer.data])
            self._mouse_down = True
            yield
            if len(layer.data)>len(previous_data):
                self.bounding_box_layer.features.iloc[-1, self.bounding_box_layer.features.columns.get_loc("label")] = self.next_index()
                text = dict(layer.text)
                text["text"] = "{label:d}"
                layer.text = text
                layer.refresh_text()
            while event.type == "mouse_move":
                yield
            self._mouse_down = False
            new_data = np.asarray(layer.data)
            # the bounding boxes are in world coordinates
            if self.image_layer:
                im_size = self.image_layer.data.shape
                if self.image_layer.rgb:
                    im_size = im_size[:-1]
                world_min = np.asarray(self.image_layer.translate)
                world_max = np.asarray(im_size)*self.image_layer.scale + world_min
            else:
                world_min = np.zeros(new_data.shape[-1])
                world_max = np.full(new_data.shape[-1], np.inf)
            if len(new_data) > 0 and (np.any(np.all(new_data > world_max, 1))\
                    or np.any(np.all(new_data < world_min, 1))):
                layer.data = previous_data
            if len(new_data) > len(previous_data):
                layer.data[-1][:] = np.clip(layer.data[-1], world_min, world_max - self.image_layer.scale)
                layer.data = layer.data
            self.update_items()
            if self.projections_widget is not None:
                for p in self.projections_widget.projections:
                    p.update_overlay()
                    p.update()

        def update_items(self, update_all_icons=False):
            bounding_boxes = self.bounding_box_corners()
            if len(bounding_boxes) == 0:
                self.clear()
            elif len(bounding_boxes) < self.count():
                previous_data = np.asarray(list(np.unique(self.item(i).bounding_box, axis=0) for i in range(self.count())))
                removed_idx = np.argwhere(np.all(np.any(~np.equal(previous_data[np.newaxis], bounding_boxes[:, np.newaxis]), (-2, -1)), axis=0))
                removed_idx = np.squeeze(removed_idx)
                if removed_idx.ndim == 0:
                    self.takeItem(removed_idx)
                else:
                    self.clear()
            for i, bb in enumerate(bounding_boxes):
                curr_item = self.item(i)
                if curr_item is not None:
                    if not np.allclose(curr_item.bounding_box, bb):
                        curr_item.bounding_box = bb
                    if curr_item.image_layer is not self.image_layer:
                        curr_item.image_layer = self.image_layer
                    if curr_item.idx != self.bounding_box_layer.features["label"][i]:
                        curr_item.idx = self.bounding_box_layer.features["label"][i]
                    if update_all_icons:
                        curr_item.update_icon()

                else:
                    new_item = QObjectListWidgetItem(self.name_template, bb, self.bounding_box_layer.features["label"][i], self, self.viewer, self.image_layer, self.mask_layer, self.channels_dim)
                    self.insertItem(i, new_item)

        def insertItem(self, index, item: QObjectListWidgetItem):
            super().insertItem(index, item)
            self.setItemWidget(item, item.object_item)

        def addItem(self, item: QObjectListWidgetItem):
            self.insertItem(self.count(), item)

        def next_index(self):
            return next(self.indices)

        def _on_bb_double_click(self, layer=None, event=None):
            if len(self.bounding_box_layer.selected_data) == 0:
                return
            idx = next(iter(self.bounding_box_layer.selected_data))
            item = self.item(idx)
            self.setCurrentItem(item)
            self.select_item(item)


    class ListWidgetBB(WidgetWithLayerList):
        def __init__(self, viewer: Viewer):
            viewer = unwrap_viewer(viewer)
            super().__init__(viewer, [("bounding_box", BoundingBoxLayer), ("image", Image), ("labels", Labels)], add_layers=False)
            layout = QVBoxLayout()
            self.viewer = viewer
            self.prev_n_layers = len(viewer.layers)
            self.channels_dim = None
            self.list_widget = None
            self.reset_index()
            self.bounding_box.combobox.currentIndexChanged.connect(self.bb_index_change)
            self.image.combobox.currentIndexChanged.connect(self.img_index_change)
            self.labels.combobox.currentIndexChanged.connect(self.mask_index_change)
            layout.addWidget(self.bounding_box.combobox)
            layout.addWidget(self.image.combobox)
            layout.addWidget(self.labels.combobox)

            self.list_widget_container = QStackedWidget()
            no_data_widget = QWidget()
            no_data_layout = QVBoxLayout()
            no_data_layout.addWidget(QLabel("No bounding box layer selected"))
            add_bb_button = QPushButton("Create Bounding Box layer")
            add_bb_button.clicked.connect(self.create_bb_layer)
            no_data_layout.addWidget(add_bb_button)
            no_data_widget.setLayout(no_data_layout)
            self.list_widget_container.addWidget(no_data_widget)
            self.list_widget_container.setCurrentIndex(0)
            layout.addWidget(self.list_widget_container)
            surroundings_tooltip = ("In the 3D view the selected object is shown by clipping the original layer to its"
                                    " bounding box. Show the rest of the layer with a low opacity as well.")
            self.show_image_surroundings_checkbox = QCheckBox("Show image surroundings in 3D")
            self.show_image_surroundings_checkbox.setChecked(True)
            self.show_image_surroundings_checkbox.setToolTip(surroundings_tooltip)
            self.show_image_surroundings_checkbox.toggled.connect(self._on_surroundings_toggled)
            self.show_labels_surroundings_checkbox = QCheckBox("Show labels surroundings in 3D")
            self.show_labels_surroundings_checkbox.setToolTip(surroundings_tooltip)
            self.show_labels_surroundings_checkbox.toggled.connect(self._on_surroundings_toggled)
            layout.addWidget(self.show_image_surroundings_checkbox)
            layout.addWidget(self.show_labels_surroundings_checkbox)
            buttons_widget = QWidget()
            buttons_layout = QHBoxLayout()
            import_button = QPushButton("Import")
            import_button.clicked.connect(self.import_bounding_boxes)
            export_button = QPushButton("Export")
            export_button.clicked.connect(self.export_bounding_boxes)
            from_labels_button = QPushButton("From labels")
            from_labels_button.clicked.connect(self.bounding_boxes_from_labels)
            buttons_layout.addWidget(import_button)
            buttons_layout.addWidget(export_button)
            buttons_layout.addWidget(from_labels_button)
            buttons_widget.setLayout(buttons_layout)
            layout.addWidget(buttons_widget)
            self.setLayout(layout)
            self.name_template = "Object"
            if self.bounding_box.layer is not None:
                self.create_list_widget()
            self.installEventFilter(self)

        def _on_surroundings_toggled(self, _=None):
            if self.list_widget is not None:
                self.list_widget.show_image_surroundings = self.show_image_surroundings_checkbox.isChecked()
                self.list_widget.show_labels_surroundings = self.show_labels_surroundings_checkbox.isChecked()

        def bb_index_change(self, _=None):
            if self.bounding_box.layer is None:
                self.reset_index()
                self.remove_list_widget()
                return
            if self.list_widget is None:
                self.create_list_widget()
            self.list_widget.bounding_box_layer = self.bounding_box.layer
            self.reset_index_after_existing_labels()

        def reset_index_after_existing_labels(self):
            if "label" in self.bounding_box.layer.features and len(self.bounding_box.layer.features):
                self.reset_index(max(self.bounding_box.layer.features["label"]) + 1)
            else:
                self.reset_index()

        def mask_index_change(self, _=None):
            if self.list_widget is not None and self.list_widget.mask_layer is not self.labels.layer:
                self.list_widget.mask_layer = self.labels.layer

        def img_index_change(self, _=None):
            self.update_channels_dim()

        @property
        def image_layer(self):
            return self.image.layer

        @property
        def mask_layer(self):
            return self.labels.layer

        def create_bb_layer(self, _):
            if self.image_layer is not None:
                self.viewer.add_layer(
                    BoundingBoxLayer(ndim=self.image_layer.ndim, edge_color="green", face_color="transparent"))

        def update_channels_dim(self):
            if self.image_layer is None:
                return
            smallest_dim = np.argsort(self.image_layer.data.shape)[0]
            if self.image_layer.data.shape[smallest_dim] <= 3:
                self.channels_dim = smallest_dim
            else:
                self.channels_dim = None
            if self.list_widget is not None:
                self.list_widget.image_layer = self.image_layer
                self.list_widget.channels_dim = self.channels_dim

        def create_list_widget(self):
            self.list_widget = ObjectListWidget(self.viewer, self.name_template, self.bounding_box.layer, self.image_layer, self.labels.layer, self.channels_dim, 1,
                                                 self.show_image_surroundings_checkbox.isChecked(),
                                                 self.show_labels_surroundings_checkbox.isChecked())
            self.list_widget_container.addWidget(self.list_widget)
            self.list_widget_container.setCurrentIndex(1)
            self.reset_index_after_existing_labels()

        def remove_list_widget(self):
            if self.list_widget is not None:
                self.list_widget.release_clipper()
            while self.list_widget_container.count() > 1:
                self.list_widget_container.removeWidget(self.list_widget_container.widget(1))
            self.list_widget_container.setCurrentIndex(0)
            self.list_widget = None

        def export_bounding_boxes(self):
            if self.bounding_box.layer is None:
                return
            filename = QFileDialog.getSaveFileName(self, "Select...", None, "(*.txt)")[0]
            bboxes = np.asarray(list(map(lambda bb: np.concatenate([bb.min(axis=0), bb.max(axis=0)]), self.bounding_box.layer.data)))
            with open(filename, "w") as f:
                writer = csv.writer(f)
                for i, bb in enumerate(bboxes):
                    item = self.list_widget.item(i)
                    writer.writerow([item.name, item.idx] + list(bb))

        def import_bounding_boxes(self):
            filename = QFileDialog.getOpenFileName(self, "Select...", None, "(*.txt)")[0]
            if not filename:
                return
            names = []
            idxs = []
            data = []
            with open(filename, "r") as f:
                reader = csv.reader(f)
                for line in reader:
                    if len(line) == 0:
                        continue
                    names.append(line[0])
                    idxs.append(int(line[1]))
                    data.append(list(map(lambda x: float(x), line[2:])))
            data = np.asarray(data)
            bounding_box_corners = np.reshape(data, (len(data), 2, -1))
            mask = np.asarray(list(itertools.product((False, True), repeat=bounding_box_corners[0].shape[1])))
            bounding_boxes = np.asarray([np.where(mask, bbc[1], bbc[0]) for bbc in bounding_box_corners])
            bounding_box_layer = BoundingBoxLayer(bounding_boxes, name=os.path.basename(filename), edge_color="green", face_color="transparent", features={"label": np.asarray(idxs, dtype=int)})
            self.viewer.add_layer(bounding_box_layer)
            self.bounding_box.layer = bounding_box_layer

            for i, (name, idx) in enumerate(zip(names, idxs)):
                item = self.list_widget.item(i)
                item.name = name
                item.idx = idx
            self.reset_index(max(idxs) + 1)

        def bounding_boxes_from_labels(self):
            if self.labels.layer is None:
                return
            bb_layer = BoundingBoxLayer(ndim=self.image_layer.ndim, edge_color="green", face_color="transparent")
            bb_corners = find_objects(self.labels.layer.data)
            ids = []
            bbs = []
            for i, bb in enumerate(bb_corners):
                if bb is None:
                    continue
                min_ = [slice_.start for slice_ in bb]
                max_ = [slice_.stop - 1 for slice_ in bb]
                bb = np.asarray(np.where(list(itertools.product((False, True), repeat=self.image_layer.ndim)), max_, min_))
                bbs.append(bb)
                ids.append(i+1)
            bb_layer.data = bbs
            bb_layer.features["label"] = np.asarray(ids, dtype=int)
            self.viewer.add_layer(bb_layer)
            self.bounding_box.layer = bb_layer

        def reset_index(self, starting_index=1):
            if self.list_widget is not None:
                self.list_widget.indices = starting_index
else:
    class ListWidgetBB(QWidget):
        def __init__(self, *, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout()
            layout.addWidget(QLabel("In order to use this widget, install the 'napari-bbox' plugin."))
            self.setLayout(layout)

