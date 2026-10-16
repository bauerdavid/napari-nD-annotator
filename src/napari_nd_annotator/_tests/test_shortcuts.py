"""The "?" help dialog, the README table and the key bindings all come from one registry
(``_widgets/_utils/shortcuts.py``); these tests keep it that way.

Like ``test_napari_compat.py`` they need a viewer, so run them under ``xvfb-run`` with a software GL driver.
"""
import ast
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtWidgets import QTableWidget

import napari_nd_annotator
from napari_nd_annotator._widgets._utils import shortcuts as registry
from napari_nd_annotator._widgets._utils.shortcuts import LABELS, SCRIPT, SHORTCUTS, VIEWER, display_key

# the fixtures of the compatibility tests: a viewer, and the Annotation Toolbox docked in it
from .test_napari_compat import (  # noqa: F401
    viewer, deprecations, annotator, no_persisted_settings, _process_events, PLUGIN)


def _norm(key):
    from napari.utils.key_bindings import coerce_keybinding
    return str(coerce_keybinding(key))


def _available(shortcut):
    import importlib.util
    return not shortcut.requires or importlib.util.find_spec(shortcut.requires) is not None


def _bound_keys(target):
    return {_norm(k) for k in target.keymap}


def _registry_keys(scope):
    return {_norm(k): s for s in SHORTCUTS if s.scope == scope for k in s.keys}


def _open_object_list(viewer):
    """The Object List binds its shortcut once it has a bounding box layer to list."""
    from napari_bbox import BoundingBoxLayer
    viewer.add_layer(BoundingBoxLayer(ndim=3, name="bboxes"))
    viewer.window.add_plugin_dock_widget(PLUGIN, "Object List")
    _process_events()


@pytest.fixture
def plugin_setup(viewer, annotator, no_persisted_settings):
    """A viewer with every widget that binds keys opened, and an active labels layer the plugin set up.

    Returns the viewer, the labels layer, and the keys napari itself binds on them (what the plugin added is
    whatever is bound on top of that).
    """
    from napari.components import ViewerModel
    from napari.layers import Labels
    baseline_layer = set(Labels(np.zeros((2, 4, 4), dtype=np.uint16)).keymap)
    baseline_viewer = set(ViewerModel().keymap)
    image = viewer.add_image(np.zeros((5, 20, 20)), name="image")
    labels = viewer.add_labels(np.zeros((5, 20, 20), dtype=np.uint16), name="labels",
                               metadata={"source_image_layer": image})
    if _available(registry.BY_ID["toggle_bounding_boxes"]):
        _open_object_list(viewer)
    viewer.layers.selection.active = labels
    _process_events()
    return viewer, labels, baseline_layer, baseline_viewer


def _dialog_rows(parent):
    from napari_nd_annotator._widgets._utils.help_dialog import HelpDialog
    dialog = HelpDialog(parent)  # the table is deleted with the dialog
    table = dialog.findChild(QTableWidget)
    return [(table.item(i, 0).text(), table.item(i, 1).text()) for i in range(table.rowCount())]


def test_every_key_the_plugin_binds_is_in_the_dialog(plugin_setup):
    viewer, labels, baseline_layer, baseline_viewer = plugin_setup
    added = {
        LABELS: {_norm(k) for k in set(labels.keymap) - baseline_layer},
        VIEWER: {_norm(k) for k in set(viewer.keymap) - baseline_viewer},
    }
    assert added[LABELS] and added[VIEWER], "the plugin bound nothing; is the fixture wrong?"
    for scope, keys in added.items():
        documented = set(_registry_keys(scope))
        if not keys:
            continue
        assert keys <= documented, "bound but missing from the help dialog (%s): %s" % (scope, sorted(keys - documented))
    # ... and, since the dialog is built from the registry, those rows really are in it
    text = " ".join(shortcut for _, shortcut in _dialog_rows(None))
    for scope in (LABELS, VIEWER):
        for key in added[scope]:
            s = _registry_keys(scope)[key]
            assert display_key(s.keys[0]) in text


def test_action_manager_shortcuts_added_by_the_plugin_are_documented(plugin_setup):
    from napari.utils.action_manager import action_manager
    names = [n for n in action_manager._actions if n.startswith(PLUGIN + ":")]
    assert names, "the plugin registers no actions; is the fixture wrong?"
    documented = {_norm(k) for s in SHORTCUTS for k in s.keys}
    for name in names:
        for key in action_manager._shortcuts.get(name) or []:
            assert _norm(key) in documented, "%s is bound to %s, which the help dialog does not list" % (name, key)


def test_every_dialog_entry_is_actually_bound(plugin_setup):
    viewer, labels, *_ = plugin_setup
    rows = _dialog_rows(None)
    assert len(rows) == len(SHORTCUTS)
    by_display = {s.display: s for s in SHORTCUTS}
    bound = {LABELS: _bound_keys(labels), VIEWER: _bound_keys(viewer)}
    for _, shortcut_text in rows:
        s = by_display[shortcut_text]
        if s.scope == SCRIPT:
            continue  # bound by magicclass on the script editor widget, checked below
        if not _available(s):
            continue
        missing = [k for k in s.keys if _norm(k) not in bound[s.scope]]
        assert not missing, "shown in the help dialog but not bound on the %s: %s" % (s.scope, missing)


def test_script_editor_shortcuts_are_bound():
    from napari_nd_annotator._widgets._utils.image_processing_widget import ScriptExecuteWidget
    bound = {getattr(f, "__name__", None) for f in vars(ScriptExecuteWidget).values()}
    assert {"Run", "_increase_font_size", "_decrease_font_size"} <= bound
    assert [s.id for s in SHORTCUTS if s.scope == SCRIPT] == ["run_script", "script_font_larger", "script_font_smaller"]


def test_dialog_matches_registry():
    assert [a for a, _ in _dialog_rows(None)] == [s.description for s in SHORTCUTS]


@pytest.mark.parametrize("key, expected", [
    ("Control-+", "Ctrl+Plus"), ("Control--", "Ctrl+Minus"), ("Control-1", "Ctrl+1"), ("Control", "Ctrl"),
    ("I", "I"), ("0", "0"), ("Escape", "Escape"), ("d", "D"), ("Ctrl+Enter", "Ctrl+Enter"),
])
def test_display_key(key, expected):
    assert display_key(key) == expected


def test_registry_is_consistent():
    assert len({s.id for s in SHORTCUTS}) == len(SHORTCUTS)
    per_scope = {}
    for s in SHORTCUTS:
        for k in s.keys:
            assert (s.scope, _norm(k)) not in per_scope or s.scope == SCRIPT, "%s bound twice" % k
            per_scope[(s.scope, _norm(k))] = s


# --- nothing may bypass the registry ----------------------------------------------------

def _plugin_sources():
    root = Path(napari_nd_annotator.__file__).parent
    return [p for p in root.rglob("*.py") if "_tests" not in p.parts and p.name != "shortcuts.py"]


def test_no_bind_key_outside_the_registry():
    """New shortcuts must be declared in ``shortcuts.py``, or they would be missing from the help dialog."""
    offenders = []
    for path in _plugin_sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # layer.bind_key(...) / viewer.bind_key(...)
            if isinstance(func, ast.Attribute) and func.attr == "bind_key":
                offenders.append("%s:%d" % (path.name, node.lineno))
            # magicclass's @bind_key(...) decorator must take its key from the registry, not a literal
            elif isinstance(func, ast.Name) and func.id == "bind_key" and node.args \
                    and isinstance(node.args[0], ast.Constant):
                offenders.append("%s:%d" % (path.name, node.lineno))
    assert not offenders, "declare these shortcuts in _widgets/_utils/shortcuts.py instead: " + ", ".join(offenders)


# --- README ------------------------------------------------------------------------------

def test_readme_table_is_generated_from_the_registry():
    for readme in (Path(napari_nd_annotator.__file__).parents[2] / "README.md", Path.cwd() / "README.md"):
        if readme.exists():
            break
    else:
        pytest.skip("README.md is not next to the package")
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- shortcuts:start -->", "<!-- shortcuts:end -->"
    assert start in text and end in text
    current = text.split(start)[1].split(end)[0].strip()
    assert current == registry.markdown_table(), (
        "README shortcut table is stale; regenerate with "
        "python -m napari_nd_annotator._widgets._utils.shortcuts")
