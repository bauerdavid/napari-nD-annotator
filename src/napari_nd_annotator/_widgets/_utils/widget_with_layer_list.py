import napari
from qtpy.QtWidgets import QWidget, QComboBox, QVBoxLayout, QSizePolicy, QScrollArea
from qtpy.QtCore import Qt
from keyword import iskeyword

from napari_nd_annotator._widgets._utils.persistence import PersistentWidget


class WidgetWithLayerList(PersistentWidget):
    def __init__(self, viewer: napari.Viewer, layers, persistence_id=None, add_layers=True, scrollable=True, **kwargs):
        super().__init__(persistence_id, **kwargs)
        layout = QVBoxLayout(self)
        layers_layout = QVBoxLayout()
        if scrollable:
            self.scroll_area = QScrollArea(self)
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.central_widget = QWidget(self)
        self.viewer = viewer
        self.layers = dict()  # str -> QComboBox
        for layer in layers:  # every layer should be a tuple: (layer_name, layer_type) or (layer_name, layer_type, layer_display_name)
            if len(layer) == 2:
                layer_name, layer_type = layer
                layer_displayed_name = layer_name.replace("_", " ")
            elif len(layer) == 3:
                layer_name, layer_type, layer_displayed_name = layer
            else:
                raise IndexError("every layer should be a tuple: (layer_name, layer_type)"
                                 " or (layer_name, layer_type, layer_display_name)")
            if not layer_name.isidentifier() or iskeyword(layer_name):
                raise ValueError("layer name '%s' is not a valid attribute name (cannot be accessed as 'obj.%s')" % (layer_name, layer_name))
            self.layers[layer_name] = self.LayerRecord(self, layer_type, layer_displayed_name)
            if add_layers:
                layers_layout.addWidget(self.layers[layer_name].combobox)
        layout.addLayout(layers_layout)
        if scrollable:
            self.scroll_area.setWidget(self.central_widget)
            layout.addWidget(self.scroll_area)
        else:
            layout.addWidget(self.central_widget)

    def setLayout(self, a0: 'QLayout') -> None:
        self.central_widget.setLayout(a0)

    def __getattr__(self, item):
        if item in self.layers:
            return self.layers[item]
        raise AttributeError("No attribute named %s" % item)

    class LayerRecord:
        def __init__(self, parent, layer_type, display_name="Select a Layer"):
            self.parent = parent
            self.layer_type = layer_type
            self.combobox = QComboBox()
            self.display_name = display_name
            self.combobox.addItem("[%s]" % display_name)
            self.combobox.currentIndexChanged.connect(self.on_layer_index_change)
            self.combobox.setToolTip(display_name)
            self.combobox.setSizePolicy(QSizePolicy.Ignored, self.combobox.sizePolicy().verticalPolicy())
            self.viewer.layers.events.connect(self.on_layer_list_change)
            self._moved_layer = None
            self._layer_name = None
            self.on_layer_list_change()

        @property
        def viewer(self) -> napari.Viewer:
            return self.parent.viewer

        @property
        def layer(self):
            return self.viewer.layers[self._layer_name] if self._layer_name and self._layer_name in self.viewer.layers\
                else None

        @layer.setter
        def layer(self, layer):
            if layer is None:
                self._layer_name = None
                self.combobox.setCurrentIndex(0)
                return
            if isinstance(layer, self.layer_type):
                new_layer_name = layer.name
            elif type(layer) is str:
                new_layer_name = layer
            else:
                raise TypeError("layer should be %s or str" % self.layer_type)
            if new_layer_name == self._layer_name:
                return
            if new_layer_name in self.viewer.layers:
                self.combobox.setCurrentText(new_layer_name)
                self._layer_name = new_layer_name

        def on_layer_index_change(self, index):
            if index == 0:
                if self.combobox.count() > 1:
                    self.combobox.setCurrentText(self._layer_name)
                    return
            elif index > 0:
                self._layer_name = self.combobox.itemText(index)

        def on_layer_list_change(self, event=None):
            type_ = event.type if event else None
            if type_ not in ["moved", "inserted", "removed", "name", None]:
                return
            filtered = list(filter(lambda layer: isinstance(layer, self.layer_type), self.viewer.layers))
            if len(filtered) == 0 and type_ != "removed":
                return
            if type_ in ["moved", "inserted", "removed", None]:
                if self.combobox.count() == len(filtered) and \
                        all((layer.name == self.combobox.itemText(i+1) for i, layer in enumerate(filtered))):
                    return
                self.combobox.blockSignals(True)
                self.combobox.clear()
                self.combobox.addItem("[%s]" % self.display_name)
                for layer in filtered:
                    self.combobox.addItem(layer.name)
                    if layer.name == self._layer_name:
                        self.combobox.setCurrentText(layer.name)
                self.combobox.blockSignals(False)
                if self._layer_name != self.combobox.currentText():
                    self.combobox.currentTextChanged.emit(self.combobox.currentText())
                    self.combobox.currentIndexChanged.emit(self.combobox.currentIndex())
                if self.combobox.count() > 1 and (self.combobox.currentIndex() == 0 or self.layer not in filtered):
                    self.layer = filtered[0]
                elif self.combobox.count() == 1:
                    self.layer = None
            elif type_ == "name":
                self.combobox.blockSignals(True)
                for i in range(len(filtered)):
                    if self.combobox.itemText(i+1) != filtered[i].name:
                        if self.combobox.itemText(i+1) == self._layer_name:
                            self._layer_name = filtered[i].name
                        self.combobox.setItemText(i+1, filtered[i].name)
                        break
                self.combobox.blockSignals(False)
