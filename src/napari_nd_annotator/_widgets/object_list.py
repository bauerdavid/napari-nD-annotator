import cv2
import numpy as np

from qtpy.QtCore import QEvent, Qt, QObject
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QListWidgetItem, QListWidget, QMenu, \
    QComboBox, QPushButton, QStackedWidget, QDockWidget, QFileDialog, QSpinBox
from napari import Viewer
from napari.layers import Image, Labels
from napari.layers.labels._labels_constants import Mode
from scipy.ndimage import find_objects

from .projections import SliceDisplayWidget
from ..boundingbox import BoundingBoxLayer

import itertools
import csv

import PIL.Image as PilImage


def slice_len(slice_, count=None):
    if count is None:
        count = slice_.stop
    return len(range(*slice_.indices(count)))


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

    def bounding_box_str(self, bb):
        pass


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
        return int(self.object_item.id_label.text().split(":")[-1])

    @idx.setter
    def idx(self, idx):
        self.object_item.id_label.setText("Id: " + str(idx))

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
            self.parent.crop_image_layer.translate = tuple(self.bounding_box[0, :])
            self.parent.crop_image_layer.refresh()
            self.parent.crop_mask_layer.data = self.mask_layer.data[self.bbox_idx]
            self.parent.crop_mask_layer.translate = tuple(self.bounding_box[0, :])
            self.parent.crop_mask_layer.refresh()
        self.update_icon()

    @property
    def bbox_idx(self):
        bbox_idx = tuple(
            slice(self.bounding_box[0, d], self.bounding_box[1, d]) for d in range(self.bounding_box.shape[1]))
        return bbox_idx

    @property
    def name(self):
        return self.object_item.name

    @name.setter
    def name(self, name):
        self.object_item.name = name

    def create_layers(self, colormap):
        image_layer = Image(self.image_layer.data[self.bbox_idx], name="Cropped Image %d" % self.idx, colormap=colormap,
                            rgb=self.image_layer.rgb)
        image_layer.translate = self.bounding_box[0, :]+self.image_layer.translate
        self.viewer.add_layer(image_layer)
        mask_layer = Labels(self.mask_layer.data[self.bbox_idx], name="Cropped Mask %d" % self.idx)
        mask_layer.selected_label = self.idx
        mask_layer.translate = self.bounding_box[0, :]+self.mask_layer.translate
        mask_layer.brush_size = 1
        mask_layer.mode = Mode.PAINT
        self.viewer.add_layer(mask_layer)
        return image_layer, mask_layer

    def unselect(self):
        if self.parent.crop_image_layer is not None:
            if self.parent.crop_image_layer in self.viewer.layers:
                self.viewer.layers.remove(self.parent.crop_image_layer)
            self.parent.crop_image_layer = None
        if self.parent.crop_mask_layer is not None:
            if self.parent.crop_mask_layer in self.viewer.layers:
                self.viewer.layers.remove(self.parent.crop_mask_layer)
            self.parent.crop_mask_layer.mouse_drag_callbacks.remove(QObjectListWidgetItem.update_layer)
            self.parent.crop_mask_layer = None
        if self.parent.projections_widget is not None:
            self.viewer.window.remove_dock_widget(self.parent.projections_widget)
            self.parent.projections_widget = None
        self.mask_layer.visible = True
        self.image_layer.visible = True

    def on_select(self):
        self.unselect()
        self.parent.crop_image_layer, self.parent.crop_mask_layer = self.create_layers(self.image_layer.colormap)
        self.parent.crop_image_layer.name = self.parent.crop_image_layer.name
        self.parent.crop_mask_layer.name = self.parent.crop_mask_layer.name
        self.parent.crop_mask_layer.events.set_data.connect(self.on_data_change)
        if self.image_layer.ndim > 2:
            self.parent.projections_widget = SliceDisplayWidget(self.viewer, self.parent.crop_image_layer, self.parent.crop_mask_layer, self.channels_dim)
            qt_widget = self.viewer.window.add_dock_widget(self.parent.projections_widget)
            qt_widget.setFloating(True)
            qt_widget.setFeatures(qt_widget.features() & ~QDockWidget.DockWidgetClosable)
            def set_dockable(state):
                qt_widget.setAllowedAreas(Qt.AllDockWidgetAreas if state else Qt.NoDockWidgetArea)
            self.parent.projections_widget.dockable_checkbox.clicked.connect(set_dockable)
        self.parent.crop_mask_layer.mouse_drag_callbacks.append(self.update_layer)
        self.viewer.layers.selection.select_only(self.parent.crop_mask_layer)
        self.mask_layer.visible = False
        self.image_layer.visible = False

    def on_data_change(self, event):
        if self.channels_dim is not None and not self.image_layer.rgb:
            if self.parent.crop_mask_layer._mode == Mode.PAINT:
                data = self.parent.crop_mask_layer.data.max(axis=self.channels_dim, keepdims=True)
                self.parent.crop_mask_layer.data[:] = data
            elif self.parent.crop_mask_layer._mode == Mode.ERASE:
                data = self.parent.crop_mask_layer.data.min(axis=self.channels_dim, keepdims=True)
                self.parent.crop_mask_layer.data[:] = data
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
            slice(max(self.bounding_box[0, d], 0), self.bounding_box[1, d]) if d in visible_dims
            else self.bounding_box[:, d].mean().astype(int)
            for d in range(self.bounding_box.shape[1])
        )
        if self.image_layer is None:
            icon = np.zeros([len(bbox_idx[d]) for d in range(len(bbox_idx)) if type(bbox_idx[d]) is slice])
        elif self.image_layer.dtype == np.uint16:
            icon = (self.image_layer.data[bbox_idx] // (2 ** 8)).astype(np.uint8)
        else:
            icon = self.image_layer.data[bbox_idx].astype(np.uint8)
        if self.image_layer is not None and not self.image_layer.rgb:
            icon = (self.image_layer.colormap.colors[icon] * 255).astype(np.uint8)[..., :3]
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
    def __init__(self, viewer, name_template, bounding_box_layer, image_layer, mask_layer, channels_dim, indices=None):
        super().__init__()
        self._bounding_box_layer = None
        self._image_layer = None
        self._mask_layer = None
        self.crop_image_layer = None
        self.crop_mask_layer = None
        self.projections_widget = None
        self._mouse_down = False
        self.selected_idx = None
        self.previously_hovered = None
        self.previous_face_color = None
        self.previous_edge_color = None
        self.bb_was_visible = None
        if indices is None:
            indices = range(1, 1000000)
        self.indices = itertools.cycle(indices)
        self.current_index = 0
        self.object_counter = 0
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
        self.viewer.bind_key('d')(self.toggle_bb_visibility)
        # bounding_box_layer.events.set_data.connect(on_data_change)

    def __del__(self):
        self._bounding_box_layer.mouse_drag_callbacks.remove(self.bounding_box_change)

    @property
    def bounding_box_layer(self):
        return self._bounding_box_layer

    @bounding_box_layer.setter
    def bounding_box_layer(self, new_layer):
        if self._bounding_box_layer:
            self._bounding_box_layer.mouse_drag_callbacks.remove(self.bounding_box_change)
            self._bounding_box_layer.mouse_double_click_callbacks.remove(self._on_bb_double_click)
            self._bounding_box_layer.events.disconnect(self.on_layer_event)
        self._bounding_box_layer = new_layer
        self._bounding_box_layer.mouse_drag_callbacks.append(self.bounding_box_change)
        self._bounding_box_layer.mouse_double_click_callbacks.append(self._on_bb_double_click)
        self._bounding_box_layer.events.connect(self.on_layer_event)
        self._bounding_box_layer.current_properties |= {"label": self.current_index}
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
        self._image_layer = new_layer
        for i in range(self.count()):
            item = self.item(i)
            item.image_layer = new_layer
            item.update_icon()
        self.update_items()

    @property
    def mask_layer(self):
        return self._mask_layer

    @mask_layer.setter
    def mask_layer(self, new_layer):
        self._mask_layer = new_layer
        for i in range(self.count()):
            item = self.item(i)
            item.mask_layer = new_layer
            item.update_icon()
            if self.currentIndex().row() == i:
                item.on_select()
        self.update_items()

    def on_layer_event(self, event):
        # print(event.type, self._mouse_down)
        if event.type == "set_data" and not self._mouse_down:
            self.update_items(True)

    def select_item(self, item):
        item.on_select()
        selected_idx = self.indexFromItem(item).row()
        if selected_idx == self.selected_idx:
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

    def bounding_box_corners(self):
        if self._bounding_box_layer is None or len(self._bounding_box_layer.data) == 0:
            return []
        b_data = np.round(np.asarray(self._bounding_box_layer.data)).astype(int)
        if self.image_layer is not None:
            b_data -= self.image_layer.translate.astype(int)
        return np.concatenate([b_data.min(1, keepdims=True), b_data.max(1, keepdims=True)], axis=1)

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
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
        elif event.type() == QEvent.Enter and type(source) == QObjectWidget:
            item = source.parent
            if item is None:
                return False
            idx = self.indexFromItem(item).row()
            self.previously_hovered = idx
            self.previous_face_color = self.bounding_box_layer.face_color[idx].copy()
            self.bounding_box_layer.face_color[idx] = (1., 1.0, 1.0, 0.5)
            self.bounding_box_layer.data = self.bounding_box_layer.data
            return True
        elif event.type() == QEvent.Leave and type(source) == QObjectWidget:
            if self.previous_face_color is not None:
                self.bounding_box_layer.face_color[self.previously_hovered] = self.previous_face_color
                self.bounding_box_layer.data = self.bounding_box_layer.data
                self.previously_hovered = None
                self.previous_face_color = None
            return True
        return super().eventFilter(source, event)

    def toggle_bb_visibility(self, _=None):
        if self.bounding_box_layer is not None:
            self.bounding_box_layer.visible = not self.bounding_box_layer.visible

    def bounding_box_change(self, layer, event):
        previous_data = np.asarray([bb.copy() for bb in self.bounding_box_layer.data])
        self._mouse_down = True
        print(len(previous_data), len(layer.data))
        yield
        if len(layer.data)>len(previous_data):
            self.bounding_box_layer.features["label"].iat[-1] = next(self.index())
            text = dict(self.bounding_box_layer.text)
            del text["values"]
            text["text"] = "{label:d}"
            self.bounding_box_layer.text = text
        while event.type == "mouse_move":
            yield
        self._mouse_down = False
        new_data = np.asarray(self.bounding_box_layer.data)
        if len(new_data) > 0 and (np.any(np.all(new_data > self.image_layer.data.shape, 1))\
                or np.any(np.all(new_data < 0, 1))):
            self.bounding_box_layer.data = previous_data
        if np.shape(previous_data) != np.shape(self.bounding_box_layer.data):
            if len(new_data) > len(previous_data):
                self.bounding_box_layer.data = [np.clip(data, 0, np.asarray(self.image_layer.data.shape) - 1) for data in new_data]
            else:
                removed_idx = np.argwhere(~np.any(np.equal(previous_data[np.newaxis], new_data[:, np.newaxis]), (-2, -1)))
                print("removed_idx was ", removed_idx)
                self.takeItem(removed_idx)
            '''if self.crop_mask_layer is not None:
                self.viewer.layers.remove(self.crop_mask_layer)
                self.viewer.layers.remove(self.crop_image_layer)
                self.crop_mask_layer = None
                self.crop_image_layer = None
                self.viewer.window.remove_dock_widget(self.projections_widget)'''
            self.update_items()
            if self.projections_widget is not None:
                for p in self.projections_widget.projections:
                    p.update()
        elif not np.allclose(previous_data, self.bounding_box_layer.data):
            idx = int(np.argwhere(~np.all(np.isclose(previous_data, self.bounding_box_layer.data), (1, 2))))
            item = self.item(idx)
            if item is None:
                return
            self.item(idx).bounding_box = self.bounding_box_corners()[idx]
            self.item(idx).update_icon()
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
            self.takeItem(np.squeeze(removed_idx))
        for i, bb in enumerate(bounding_boxes):
            curr_item = self.item(i)
            if curr_item is not None:
                if not np.allclose(curr_item.bounding_box, bb):
                    curr_item.bounding_box = bb
                elif curr_item.image_layer is not self.image_layer:
                    curr_item.image_layer = self.image_layer
                elif update_all_icons:
                    curr_item.update_icon()
            else:
                new_item = QObjectListWidgetItem(self.name_template, bb, self.bounding_box_layer.features["label"][i], self, self.viewer, self.image_layer, self.mask_layer, self.channels_dim)
                self.insertItem(i, new_item)

    def insertItem(self, index, item: QObjectListWidgetItem):
        super().insertItem(index, item)
        self.setItemWidget(item, item.object_item)

    def addItem(self, item: QObjectListWidgetItem):
        self.insertItem(self.count(), item)

    def index(self):
        while True:
            for idx in self.indices:
                yield idx

    def _on_bb_double_click(self, layer=None, event=None):
        if len(self.bounding_box_layer.selected_data) == 0:
            return
        idx = next(iter(self.bounding_box_layer.selected_data))
        item = self.item(idx)
        self.setCurrentItem(item)
        self.select_item(item)


class ListWidgetBB(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        layout = QVBoxLayout()
        self.viewer = viewer
        self.prev_n_layers = len(viewer.layers)
        self.channels_dim = None
        self.list_widget = None
        self.prev_bb_index = 0
        self.prev_img_index = 0
        self.prev_mask_index = 0
        viewer.layers.events.connect(self.update_layers)
        self.bounding_box_layer_dropdown = QComboBox()
        self.bounding_box_layer_dropdown.addItem("[Bounding box layer]")
        self.bounding_box_layer_dropdown.currentIndexChanged.connect(self.bb_index_change)
        self.image_layer_dropdown = QComboBox()
        self.image_layer_dropdown.addItem("[Image layer]")
        self.image_layer_dropdown.setCurrentIndex(0)
        self.image_layer_dropdown.currentIndexChanged.connect(self.img_index_change)
        self.mask_layer_dropdown = QComboBox()
        self.mask_layer_dropdown.addItem("[Label layer]")
        self.mask_layer_dropdown.setCurrentIndex(0)
        self.mask_layer_dropdown.currentIndexChanged.connect(self.mask_index_change)
        self.update_layers()
        layout.addWidget(self.bounding_box_layer_dropdown)
        layout.addWidget(self.image_layer_dropdown)
        layout.addWidget(self.mask_layer_dropdown)
        self.next_index_spinner = QSpinBox()
        self.next_index_spinner.setMinimum(1)
        self.next_index_spinner.setMaximum(1000000)
        self.next_index_spinner.setToolTip("Next index")
        self.next_index_spinner.setVisible(False)
        layout.addWidget(self.next_index_spinner)
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
        self.update_list()
        self.installEventFilter(self)

    def bb_index_change(self, index):
        if index == 0 and self.bounding_box_layer_dropdown.count() > 0:
            self.bounding_box_layer_dropdown.setCurrentIndex(self.prev_bb_index)
            return
        self.prev_bb_index = index
        self.update_list()

    def mask_index_change(self, index):
        if index == 0 and self.mask_layer_dropdown.count() > 0:
            self.mask_layer_dropdown.setCurrentIndex(self.prev_mask_index)
            return
        if self.list_widget is not None:
            self.list_widget.mask_layer = self.mask_layer
        self.prev_mask_index = index

    def img_index_change(self, index):
        if index == 0 and self.image_layer_dropdown.count() > 0:
            self.image_layer_dropdown.setCurrentIndex(self.prev_img_index)
            return
        self.prev_img_index = index
        self.update_channels_dim(index)

    def update_layers(self, event=None):
        type_ = event.type if event else "reordered"
        # print(type_)
        if type_ in ["reordered", "removed"]:
            bb_idx = 1
            img_idx = 1
            mask_idx = 1
            for layer in self.viewer.layers:
                if isinstance(layer, BoundingBoxLayer):
                    if bb_idx >= self.bounding_box_layer_dropdown.count():
                        self.bounding_box_layer_dropdown.addItem(layer.name)
                    else:
                        self.bounding_box_layer_dropdown.setItemText(bb_idx, layer.name)
                    if self.bounding_box_layer is None\
                            or self.bounding_box_layer_dropdown.itemText(bb_idx) == self.bounding_box_layer.name:
                        self.bounding_box_layer_dropdown.setCurrentIndex(bb_idx)
                    bb_idx += 1
                elif isinstance(layer, Image):
                    if img_idx >= self.image_layer_dropdown.count():
                        self.image_layer_dropdown.addItem(layer.name)
                    else:
                        self.image_layer_dropdown.setItemText(img_idx, layer.name)
                    if self.image_layer is None\
                            or self.image_layer_dropdown.itemText(img_idx) == self.image_layer.name:
                        self.image_layer_dropdown.setCurrentIndex(img_idx)
                    img_idx += 1
                elif isinstance(layer, Labels):
                    if mask_idx >=self.mask_layer_dropdown.count():
                        self.mask_layer_dropdown.addItem(layer.name)
                    else:
                        self.mask_layer_dropdown.setItemText(mask_idx, layer.name)
                    if self.mask_layer is None\
                            or self.mask_layer_dropdown.itemText(mask_idx) == self.mask_layer.name:
                        self.mask_layer_dropdown.setCurrentIndex(mask_idx)
                    mask_idx += 1
            if type_ == "removed":
                while bb_idx < self.bounding_box_layer_dropdown.count():
                    self.bounding_box_layer_dropdown.removeItem(bb_idx)
                while img_idx < self.image_layer_dropdown.count():
                    self.image_layer_dropdown.removeItem(img_idx)
                while mask_idx < self.mask_layer_dropdown.count():
                    self.mask_layer_dropdown.removeItem(mask_idx)
        elif type_ == "inserted":
            layer = self.viewer.layers[-1]
            if isinstance(layer, BoundingBoxLayer):
                self.bounding_box_layer_dropdown.addItem(layer.name)
                if self.bounding_box_layer_dropdown.count() == 2:
                    self.bounding_box_layer_dropdown.setCurrentIndex(1)
            elif isinstance(layer, Image):
                self.image_layer_dropdown.addItem(layer.name)
                if self.image_layer_dropdown.count() == 2:
                    self.image_layer_dropdown.setCurrentIndex(1)
            elif isinstance(layer, Labels):
                self.mask_layer_dropdown.addItem(layer.name)
                if self.mask_layer_dropdown.count() == 2:
                    self.mask_layer_dropdown.setCurrentIndex(1)
        elif type_ == "name":
            layer = event.source[event.index]
            if isinstance(layer, BoundingBoxLayer):
                self.bounding_box_layer_dropdown.setItemText(event.index+1, layer.name)
            elif isinstance(layer, Image):
                self.image_layer_dropdown.setItemText(event.index+1, layer.name)
            elif isinstance(layer, Labels):
                self.mask_layer_dropdown.setItemText(event.index+1, layer.name)


    @property
    def bounding_box_layer(self):
        return self.viewer.layers[self.bounding_box_layer_dropdown.currentText()] \
            if self.bounding_box_layer_dropdown.currentIndex() > 0 else None

    @property
    def image_layer(self):
        return self.viewer.layers[self.image_layer_dropdown.currentText()] \
            if self.image_layer_dropdown.currentIndex() > 0 else None

    @property
    def mask_layer(self):
        return self.viewer.layers[self.mask_layer_dropdown.currentText()] \
            if self.mask_layer_dropdown.currentIndex() > 0 else None

    def create_bb_layer(self, _):
        if self.image_layer is not None:
            self.viewer.add_layer(
                BoundingBoxLayer(ndim=self.image_layer.ndim, edge_color="green", face_color="transparent"))

    def update_channels_dim(self, idx):
        if idx == 0 or self.image_layer is None:
            return
        smallest_dim = np.argsort(self.image_layer.data.shape)[0]
        if self.image_layer.data.shape[smallest_dim] <= 3:
            self.channels_dim = smallest_dim
        else:
            self.channels_dim = None
        if self.list_widget is not None:
            self.list_widget.image_layer = self.image_layer
            self.list_widget.channels_dim = self.channels_dim

    def update_list(self):
        if self.bounding_box_layer is not None:
            self.list_widget = ObjectListWidget(self.viewer, self.name_template, self.bounding_box_layer, self.image_layer, self.mask_layer, self.channels_dim, self.index_iterator())
            self.list_widget_container.addWidget(self.list_widget)
            self.list_widget_container.setCurrentIndex(1)
        else:
            self.list_widget_container.setCurrentIndex(0)

    def export_bounding_boxes(self):
        if self.bounding_box_layer is None:
            return
        filename = QFileDialog.getSaveFileName(self, "Select...", None, "(*.txt)")[0]
        bboxes = np.asarray(list(map(lambda bb: np.concatenate([bb.min(axis=0), bb.max(axis=0)]), self.bounding_box_layer.data)))
        with open(filename, "w") as f:
            writer = csv.writer(f)
            for i, bb in enumerate(bboxes):
                item = self.list_widget.item(i)
                writer.writerow([item.name, item.idx] + list(bb))

    def import_bounding_boxes(self):
        filename = QFileDialog.getOpenFileName(self, "Select...", None, "(*.txt)")[0]
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
        if self.bounding_box_layer is None:
            bounding_box_layer = BoundingBoxLayer(bounding_boxes, edge_color="green", face_color="transparent")
            self.viewer.add_layer(bounding_box_layer)
        else:
            self.bounding_box_layer.data = bounding_boxes

        for i, (name, idx) in enumerate(zip(names, idxs)):
            item = self.list_widget.item(i)
            item.name = name
            item.idx = idx
        self.next_index_spinner.setValue(max(idxs)+1)

    def bounding_boxes_from_labels(self):
        if self.mask_layer is None:
            return
        if self.bounding_box_layer is None:
            bb_layer = BoundingBoxLayer(ndim=self.image_layer.ndim, edge_color="green", face_color="transparent")
            self.viewer.add_layer(bb_layer)
        else:
            bb_layer = self.bounding_box_layer
        bb_corners = find_objects(self.mask_layer.data)
        ids = []
        bbs = []
        for i, bb in enumerate(bb_corners):
            if bb is None:
                continue
            min_ = [slice_.start for slice_ in bb]
            max_ = [slice_.stop - 1 for slice_ in bb]
            bb = np.asarray(np.where(list(itertools.product((False, True), repeat=3)), max_, min_))
            bbs.append(bb)
            ids.append(i+1)
        bb_layer.data = bbs
        for i, id_ in enumerate(ids):
            self.list_widget.item(i).idx = id_

    def index_iterator(self):
        while True:
            idx = self.next_index_spinner.value()
            self.next_index_spinner.setValue(idx+1)
            yield idx

    def eventFilter(self, a0: 'QObject', a1: 'QEvent') -> bool:
        if a0 == self and a1.type() == QEvent.Show:
            self.parent().setAllowedAreas(Qt.LeftDockWidgetArea)
            return True
        return super().eventFilter(a0, a1)
