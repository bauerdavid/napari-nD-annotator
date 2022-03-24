import math

import cv2
import numpy as np
import typing

from PyQt5 import sip
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QListWidgetItem, QListWidget, QMenu
from napari.layers import Image, Labels
from napari.layers.labels._labels_constants import Mode
from qtpy import QtCore, QtGui

from .projections import SliceDisplayWidget


class QObjectWidget (QWidget):
    def __init__ (self, icon=None, name="Object", bounding_box=None, parent = None):
        super().__init__()
        self.textQVBoxLayout = QVBoxLayout()
        self.name = name
        self.textUpQLabel = QLineEdit()
        self.textDownQLabel = QLabel()
        self.textQVBoxLayout.addWidget(self.textUpQLabel)
        self.textQVBoxLayout.addWidget(self.textDownQLabel)
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


    def setTextUp (self, text):
        self.textUpQLabel.setText(text)

    def setTextDown (self, bb):
        self.textDownQLabel.setText(("corner 1: %s\ncorner 2: %s" % tuple(bb)))

    def setIcon(self, icon):
        self.iconQLabel.setPixmap(icon)

    def bounding_box_str(self, bb):
        pass


class QObjectListWidgetItem(QListWidgetItem):
    def __init__(self, parent, image_layer, mask_layer, channels_dim, icon=None, name="Object", bounding_box=None, *args):
        super().__init__(parent, *args)
        self.object_item = QObjectWidget(name=name, bounding_box=bounding_box, parent=self)
        self.icon = icon
        self.setSizeHint(self.object_item.sizeHint())
        self.setFlags(self.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        self.parent = parent
        self.parent.addItem(self)
        self.parent.setItemWidget(self, self.object_item)
        self.image_layer = image_layer
        self.mask_layer = mask_layer
        self.bounding_box = bounding_box
        self.idx = parent.count()
        self.channels_dim = channels_dim

    def __hash__(self):
        return id(self)

    @property
    def icon(self):
        return self._icon

    @icon.setter
    def icon(self, icon):
        self._icon = icon.reshape(icon.shape[0], icon.shape[1], -1)
        if self._icon.shape[-1] == 1:
            self._icon = np.tile(self._icon, (1, 1, 3))
        elif self._icon.shape[-1] == 4:
            self._icon = self._icon[..., :-1]
        scale = 64 / max(self._icon.shape[1], self._icon.shape[0])
        out_size = (int(self._icon.shape[1]*scale), int(self._icon.shape[0]*scale))
        self._resized_icon = cv2.resize(self._icon, out_size, interpolation=cv2.INTER_LANCZOS4)
        img = QImage(self._resized_icon, self._resized_icon.shape[1], self._resized_icon.shape[0], self._resized_icon.shape[1] * 3, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap()
        self.pixmap = self.pixmap.fromImage(img, QtCore.Qt.ImageConversionFlag.ColorOnly)
        self.object_item.setIcon(self.pixmap)

    @property
    def bounding_box(self):
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, bounding_box):
        self._bounding_box = bounding_box
        self.object_item.setTextDown(bounding_box)
        self.object_item.update()
        if self.parent.crop_image_layer is not None and \
            self.parent.selected_idx == self.parent.indexFromItem(self):
            self.parent.crop_image_layer.data = self.image_layer.data[self.bbox_idx]
            self.parent.crop_image_layer.translate = tuple(self.bounding_box[0, :])
            self.parent.crop_image_layer.refresh()
            self.parent.crop_mask_layer.data = self.mask_layer.data[self.bbox_idx]
            self.parent.crop_mask_layer.translate = tuple(self.bounding_box[0, :])
            self.parent.crop_mask_layer.refresh()

    @property
    def bbox_idx(self):
        bbox_idx = tuple(
            slice(self.bounding_box[0, d], self.bounding_box[1, d]) for d in range(self.bounding_box.shape[1]))
        return bbox_idx

    @property
    def name(self):
        return self.object_item.name

    def create_layers(self, viewer, colormap):
        image_layer = Image(self.image_layer.data[self.bbox_idx], name="Cropped Image %d" % self.idx, colormap=colormap,
                            rgb=self.image_layer.rgb)
        image_layer.translate = self.bounding_box[0, :]+self.image_layer.translate
        viewer.add_layer(image_layer)
        mask_layer = Labels(self.mask_layer.data[self.bbox_idx], name="Cropped Mask %d" % self.idx)
        mask_layer.selected_label = self.idx
        mask_layer.translate = self.bounding_box[0, :]+self.mask_layer.translate
        viewer.add_layer(mask_layer)
        return image_layer, mask_layer

    def on_double_click(self, viewer, colormap):
        if self.parent.crop_image_layer in viewer.layers:
            viewer.layers.remove(self.parent.crop_image_layer)
        if self.parent.crop_mask_layer in viewer.layers:
            viewer.layers.remove(self.parent.crop_mask_layer)
            if self.parent.crop_image_layer is not None:
                self.parent.crop_mask_layer.mouse_drag_callbacks.remove(QObjectListWidgetItem.update_layer)
        if self.parent.projections_widget is not None:
            viewer.window.remove_dock_widget(self.parent.projections_widget)

        self.parent.crop_image_layer, self.parent.crop_mask_layer = self.create_layers(viewer, colormap)
        self.parent.crop_image_layer.name = "[tmp] " + self.parent.crop_image_layer.name
        self.parent.crop_mask_layer.name = "[tmp] " + self.parent.crop_mask_layer.name
        self.parent.crop_mask_layer.events.set_data.connect(self.on_data_change)
        self.parent.projections_widget = SliceDisplayWidget(viewer, self.parent.crop_image_layer, self.parent.crop_mask_layer, self.channels_dim)
        qt_widget = viewer.window.add_dock_widget(self.parent.projections_widget)
        qt_widget.setFloating(True)
        self.parent.crop_mask_layer.mouse_drag_callbacks.append(self.update_layer)
        viewer.layers.selection.select_only(self.parent.crop_mask_layer)

    def on_data_change(self, event):
        if self.channels_dim is not None and not self.image_layer.rgb:
            if self.parent.crop_mask_layer._mode == Mode.PAINT:
                data = self.parent.crop_mask_layer.data.max(axis=self.channels_dim, keepdims=True)
                self.parent.crop_mask_layer.data[:] = data
            elif self.parent.crop_mask_layer._mode == Mode.ERASE:
                data = self.parent.crop_mask_layer.data.min(axis=self.channels_dim, keepdims=True)
                self.parent.crop_mask_layer.data[:] = data
        self.mask_layer.data[self.bbox_idx] = self.parent.crop_mask_layer.data
        self.parent.projections_widget.update_slider_ranges()
    @staticmethod
    def update_layer(layer, event):
        # print("mask click")
        yield
        while event.type == "mouse_move":
            # print("mask move")
            yield
        # print("mask release")
        layer.refresh()

def on_data_change(layer):
    print(layer)

class ListWidget(QListWidget):
    def __init__(self, viewer, name_template, bounding_box_layer, image_layer, mask_layer, channels_dim):
        self.bounding_box_layer = bounding_box_layer
        self.viewer = viewer
        self.image_layer = image_layer
        self.crop_image_layer = None
        self.crop_mask_layer = None
        self.projections_widget = None
        self.channels_dim = channels_dim
        icons = self.get_icons()
        self.name_template = name_template
        if name_template.find("#") >= 0:
            name_template = name_template.replace("#", "%.{0}d".format(int(math.log10(len(icons)) + 1)))
            names = [name_template % (i + 1) for i in range(len(icons))]
        else:
            names = [name_template for _ in range(len(icons))]
        super().__init__()
        self.setObjectName("Objects")
        assert len(icons) == len(names)
        bounding_boxes = self.bounding_box_corners()
        for icon, name, bb in zip(icons, names, bounding_boxes):
            QObjectListWidgetItem(self, image_layer, mask_layer, channels_dim, icon, name, bb)
        self.mask_layer = mask_layer
        self.itemDoubleClicked.connect(self.on_double_click)
        bounding_box_layer.mouse_drag_callbacks.append(self.bounding_box_change)
        self.selected_idx = None
        self.installEventFilter(self)
        # bounding_box_layer.events.set_data.connect(on_data_change)

    def on_double_click(self, item):
        item.on_double_click(self.viewer, self.image_layer.colormap)
        self.selected_idx = self.indexFromItem(item)

    def get_icons(self, idx=None):
        if idx is None:
            slice_idx = slice(None)
        elif type(idx) == int:
            slice_idx = slice(idx, idx+1)
        else:
            raise TypeError("idx must be of type int")
        bounding_boxes = self.bounding_box_corners()[slice_idx]
        visible_dims = list(sorted(self.viewer.dims.displayed, key=lambda d: self.image_layer.data.shape[d]))[-2:]
        bbox_idx = [tuple(
            slice(max(bounding_boxes[i, 0, d], 0), bounding_boxes[i, 1, d]) if d in visible_dims
            else bounding_boxes[i, :, d].mean().astype(int)
            for d in range(self.bounding_box_layer.ndim))
                    for i in range(len(bounding_boxes))]
        if self.image_layer.dtype == np.uint16:
            icons = [(self.image_layer.data[bbox_idx[i]] // (2 ** 8)).astype(np.uint8) for i in range(len(bbox_idx))]
        else:
            icons = [self.image_layer.data[bbox_idx[i]].astype(np.uint8) for i in range(len(bbox_idx))]
        if not self.image_layer.rgb:
            icons = [(self.image_layer.colormap.colors[icon] * 255).astype(np.uint8)[..., :3] for icon in icons]
        else:
            icons = [icon.astype(np.uint8)[..., :3] for icon in icons]

        return icons if idx is None else icons[0]

    def bounding_box_corners(self):
        b_data = (np.asarray(self.bounding_box_layer.data).round()-self.image_layer.translate).astype(int)
        return np.concatenate([b_data.min(1, keepdims=True), b_data.max(1, keepdims=True)], axis=1)

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QEvent.Close:
            if self.crop_image_layer is not None:
                self.viewer.layers.remove(self.crop_image_layer)
                self.viewer.layers.remove(self.crop_mask_layer)
                self.viewer.window.remove_dock_widget(self.projections_widget)
        elif event.type() == QEvent.ContextMenu and source is self:
            menu = QMenu()
            menu.addAction("Create layers")
            if menu.exec_(event.globalPos()):
                item = self.itemAt(event.pos())
                item.create_layers(self.viewer, self.image_layer.colormap)
            return True
        return super().eventFilter(source, event)

    def bounding_box_change(self, layer, event):
        previous_data = np.asarray([bb.copy() for bb in self.bounding_box_layer.data])
        yield
        while event.type == "mouse_move":
            yield
        new_data = np.asarray(self.bounding_box_layer.data)
        if np.any(np.all(new_data > self.image_layer.data.shape, 1))\
                or np.any(np.all(new_data < 0, 1)):
            self.bounding_box_layer.data = previous_data
        if np.shape(previous_data) != np.shape(self.bounding_box_layer.data):
            if len(new_data) > len(previous_data):
                self.bounding_box_layer.data = [np.clip(data, 0, np.asarray(self.image_layer.data.shape)-1) for data in new_data]
                names = [self.item(i).name for i in range(self.count())] + [self.item_name(self.count(), self.count()+1)]
            else:
                removed_idx = np.argwhere(~np.any(np.equal(previous_data[np.newaxis], new_data[:, np.newaxis]), (-2, -1)))
                names = [self.item(i).name for i in range(self.count()) if i != removed_idx]
            self.clear()
            if self.crop_mask_layer is not None:
                self.viewer.layers.remove(self.crop_mask_layer)
                self.viewer.layers.remove(self.crop_image_layer)
                self.crop_mask_layer = None
                self.crop_image_layer = None
                self.viewer.window.remove_dock_widget(self.projections_widget)
            for i, (icon, name, bb) in enumerate(zip(self.get_icons(), names, self.bounding_box_corners())):
                QObjectListWidgetItem(self, self.image_layer, self.mask_layer, self.channels_dim, icon, name, bb)
            if self.projections_widget is not None:
                for p in self.projections_widget.projections:
                    p.update()
        elif not np.allclose(previous_data, self.bounding_box_layer.data):
            idx = int(np.argwhere(~np.all(np.isclose(previous_data, self.bounding_box_layer.data), (1, 2))))
            self.item(idx).bounding_box = self.bounding_box_corners()[idx]
            icon = self.get_icons(idx)
            self.item(idx).icon = icon
            if self.projections_widget is not None:
                for p in self.projections_widget.projections:
                    p.update_overlay()
                    p.update()

    def item_name(self, idx, n_elements=None):
        n_elements = n_elements or self.count()
        if self.name_template.find("#") >= 0:
            name_template = self.name_template.replace("#", "%.{0}d".format(int(math.log10(n_elements) + 1)))
            return name_template % idx
        return self.name_template
