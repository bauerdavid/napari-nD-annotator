"""The settings files are written to per-user folders that may not exist yet on a fresh profile."""
import pytest
from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import QApplication

from napari_nd_annotator._widgets._utils.persistence.persistent_widget_state import PersistentWidgetState


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def isolated_state():
    """Give back the singleton in the state it was found, without letting it save to a temp folder."""
    had_instance = hasattr(PersistentWidgetState, "instance")
    previous = getattr(PersistentWidgetState, "instance", None)
    yield
    current = getattr(PersistentWidgetState, "instance", None)
    if current is not None and current is not previous:
        current.save_state = lambda: None
    if had_instance:
        PersistentWidgetState.instance = previous
    elif current is not None:
        del PersistentWidgetState.instance


def test_config_file_is_created_in_missing_folder(qapp, isolated_state, tmp_path, monkeypatch):
    config_folder = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(QStandardPaths, "writableLocation", lambda *_: str(config_folder))

    state = PersistentWidgetState()

    config_file = config_folder / "nd_annotator_config.yaml"
    assert config_file.is_file()
    assert config_file.read_text() != ""
    assert state._state  # the defaults were loaded as well


def test_state_is_saved_after_folder_was_removed(qapp, isolated_state, tmp_path, monkeypatch):
    config_folder = tmp_path / "config"
    monkeypatch.setattr(QStandardPaths, "writableLocation", lambda *_: str(config_folder))
    state = PersistentWidgetState()
    (config_folder / "nd_annotator_config.yaml").unlink()
    config_folder.rmdir()

    state.save_state()

    assert (config_folder / "nd_annotator_config.yaml").is_file()


def test_widget_with_layer_list_dumps_into_missing_folder(qapp, tmp_path, monkeypatch):
    from magicgui.widgets import Container
    from napari_nd_annotator._widgets._utils.widget_with_layer_list import WidgetWithLayerList2

    monkeypatch.setattr(
        WidgetWithLayerList2, "_dump_path", property(lambda self: tmp_path / "missing" / "cache" / "widget")
    )
    # only the path handling of ``_dump`` is of interest, so skip the viewer-dependent constructor
    widget = WidgetWithLayerList2.__new__(WidgetWithLayerList2)
    Container.__init__(widget)

    widget._dump()

    assert (tmp_path / "missing" / "cache" / "widget").is_file()
