import warnings
import os
import yaml
from qtpy.QtWidgets import QWidget
from qtpy.QtCore import QStandardPaths
from typing import Union
from traceback import print_exc

GETTER_FUN_NAMES = ["isChecked", "value", "text", "currentText"]
SETTER_FUN_NAMES = ["setChecked", "setValue", "setText", "setCurrentText"]

__location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
default_settings_path = os.path.join(__location__, "default_widget_values.yaml")


class UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError("key %s already taken (=%s)" % (key, self[key]))
        super().__setitem__(key, value)


class PersistentWidgetState:
    def __init__(self):
        config_folder = QStandardPaths.writableLocation(QStandardPaths.ConfigLocation)
        self._config_path = os.path.join(config_folder, "nd_annotator_config.yaml")
        self._state: UniqueDict = UniqueDict()
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r") as f:
                    self._state = yaml.safe_load(f)
            except Exception:
                print_exc()
        else:
            try:
                with open(self._config_path, "w") as new_file, open(default_settings_path, "r") as def_file:
                    new_file.write(def_file.read())
                    def_file.seek(0)
                    self._state = yaml.safe_load(def_file)
            except Exception:
                print_exc()

    def store_multiple_state(self, parent_name: str, widget_id_map: dict):
        widget_state = self[parent_name]
        for id_, widget in widget_id_map.items():
            self.store_state(widget_state, id_, widget)

    def store_state(self, parent: Union[str, dict], widget_id: str, widget: QWidget):
        if type(parent) is str:
            widget_state = self[parent]
        else:
            widget_state = parent
        getter_fun = None
        for fun_name in GETTER_FUN_NAMES:
            if hasattr(widget, fun_name):
                getter_fun = getattr(widget, fun_name)
                break
        if getter_fun is None:
            warnings.warn("Cannot get current value of %s (id: %s):"
                          " object type should define one of (%s)"
                          % (widget, widget_id, ", ".join(GETTER_FUN_NAMES)))
            return
        widget_state[widget_id] = getter_fun()

    def load_multiple_state(self, parent_name: str, widget_id_map: dict):
        widget_state = self._state[parent_name]
        for id_, widget in widget_id_map.items():
            self.load_state(widget_state, id_, widget)

    def load_state(self, parent: Union[str, dict], widget_id: str, widget: QWidget):
        if type(parent) is str:
            widget_state = self[parent]
        else:
            widget_state = parent
        if widget_id not in widget_state:
            warnings.warn("id '%s' not found in config file" % widget_id)
            return
        setter_fun = None
        for fun_name in SETTER_FUN_NAMES:
            if hasattr(widget, fun_name):
                setter_fun = getattr(widget, fun_name)
                break
        if setter_fun is None:
            warnings.warn("Couldn't load value for widget %s (id: %s)"
                          " object type should define one of (%s)"
                          % (widget, widget_id, ", ".join(SETTER_FUN_NAMES)))
            return
        try:
            setter_fun(widget_state[widget_id])
        except TypeError:
            print_exc()

    def __getitem__(self, item):
        if item not in self._state:
            self._state[item] = UniqueDict()
        return self._state[item]

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(PersistentWidgetState, cls).__new__(cls)
        return cls.instance

    def save_state(self):
        state = dict()
        for k, v in self._state.items():
            state[k] = dict(v)
        try:
            with open(self._config_path, "w") as f:
                yaml.dump(state, f)
        except Exception as e:
            print("Couldn't save plugin state due to the following error:")
            print_exc()

    def __del__(self):
        self.save_state()


class PersistentWidget(QWidget):
    _widget_count = 0

    def __init__(self, id_: str = None, **kwargs):
        super().__init__(**kwargs)
        self._annotator_state = PersistentWidgetState()
        self._stored_widgets = dict()
        self._id = id_
        self.destroyed.connect(lambda o: self.on_destroy(self._annotator_state, self._id, self._stored_widgets))
        PersistentWidget._widget_count += 1

    def set_stored_widgets(self, widget_ids: Union[list, dict]):
        self._stored_widgets.clear()
        if type(widget_ids) is list:
            for id_ in widget_ids:
                self._stored_widgets[id_] = getattr(self, id_)
        else:
            for id_, widget in widget_ids.items():
                self._stored_widgets[id_] = widget
        self._annotator_state.load_multiple_state(self._id, self._stored_widgets)

    def add_stored_widget(self, id_, widget=None):
        if widget is None:
            widget = getattr(self, id_)
        self._stored_widgets[id_] = widget
        self._annotator_state.load_state(self._id, id_, widget)

    @staticmethod
    def on_destroy(state, id_, widgets):
        PersistentWidget._widget_count -= 1
        if len(widgets) > 0 and id_ is not None:
            state.store_multiple_state(id_, widgets)
        if PersistentWidget._widget_count == 0:
            state.save_state()
        elif PersistentWidget._widget_count < 0:
            warnings.warn("negative 'PersistentWidget._widget_count': %d" % PersistentWidget._widget_count)
