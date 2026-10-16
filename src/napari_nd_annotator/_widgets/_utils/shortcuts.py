"""The single place where the plugin's keyboard shortcuts are declared.

Every shortcut is a :class:`Shortcut` in :data:`SHORTCUTS`. Widgets bind them with :func:`bind_all`
(they only supply the callbacks, the keys come from here), and the help dialog, the README table and the
tests are all generated from the same list, so they cannot drift apart. Calling ``.bind_key(`` anywhere
else in the package is rejected by ``test_shortcuts.py``.

Run ``python -m napari_nd_annotator._widgets._utils.shortcuts`` to print the README table.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

LABELS = "labels"  # bound on a napari Labels layer
VIEWER = "viewer"  # bound on the napari viewer
SCRIPT = "script"  # bound by magicclass on the script editor widget (``@bind_key``), see image_processing_widget

_WHEEL_UP = "Wheel ↑"
_WHEEL_DOWN = "Wheel ↓"


@dataclass(frozen=True)
class Shortcut:
    """One keyboard shortcut.

    ``keys`` are napari key strings ("Control-+"). More than one key means one binding per key, all
    described by one row (the layer jump Ctrl+1 .. Ctrl+9); the callback is then a factory taking the key.
    ``alt_keys`` are display-only alternatives that are not key bindings, e.g. the mouse wheel variants
    handled in ``callbacks.py`` / ``MinimalContourWidget._on_mouse_wheel``.
    The callback itself is supplied when binding (it depends on the widget instance), by ``id``.
    """
    id: str
    keys: Tuple[str, ...]
    description: str
    scope: str
    alt_keys: Tuple[str, ...] = ()
    # False for the bindings that must not replace one that already exists
    overwrite: bool = True
    # the alt_keys are napari's own functionality (marked in the README)
    alt_builtin: bool = False
    # importable name of the optional package the binding only exists with (tests skip it when missing)
    requires: str = ""

    @property
    def display_keys(self) -> str:
        shown = [display_key(k) for k in self.keys]
        return shown[0] if len(shown) == 1 else "%s – %s" % (shown[0], shown[-1])

    @property
    def display(self) -> str:
        """Keyboard and wheel variants as shown in the help dialog."""
        return " / ".join((self.display_keys,) + self.alt_keys)


def display_key(key: str) -> str:
    """napari key string -> display form: "Control-+" -> "Ctrl+Plus", "Control-1" -> "Ctrl+1"."""
    names = {"Control": "Ctrl", "+": "Plus", "-": "Minus"}
    parts = []
    while "-" in key[1:]:  # modifiers; a trailing "-" or "+" is the key itself
        head, key = key.split("-", 1)
        parts.append(head)
    parts.append(key)
    return "+".join(names.get(p, p.upper() if len(p) == 1 else p) for p in parts)


SHORTCUTS: Tuple[Shortcut, ...] = (
    Shortcut("increment_label", ("E",), "Increment selected label", LABELS, ("Shift+" + _WHEEL_UP,)),
    Shortcut("decrement_label", ("Q",), "Decrement selected label", LABELS, ("Shift+" + _WHEEL_DOWN,)),
    Shortcut("previous_slice", ("A",), "Previous slice", LABELS, ("Ctrl+" + _WHEEL_UP,), alt_builtin=True),
    Shortcut("next_slice", ("D",), "Next slice", LABELS, ("Ctrl+" + _WHEEL_DOWN,), alt_builtin=True),
    Shortcut("increase_brush", ("W",), "Increase paint brush size", LABELS, ("Alt+" + _WHEEL_UP,)),
    Shortcut("decrease_brush", ("S",), "Decrease paint brush size", LABELS, ("Alt+" + _WHEEL_DOWN,)),
    Shortcut("extend_mask", ("Control-+",), "Extend the selected label by one pixel (dilation)", LABELS),
    Shortcut("reduce_mask", ("Control--",), "Reduce the selected label by one pixel (erosion)", LABELS),
    Shortcut("interpolate", ("I",), "Interpolate", LABELS, overwrite=False),
    Shortcut("minimal_contour_mode", ("0",), "Switch to minimal contour mode", LABELS, overwrite=False),
    Shortcut("straight_lines", ("Control",), "Hold to draw straight lines in minimal contour mode", LABELS,
             overwrite=False),
    Shortcut("clear_contour", ("Escape",), "Clear the minimal contour being drawn", LABELS),
    Shortcut("jump_to_layer", tuple("Control-%d" % i for i in range(1, 10)), "Jump to the layer at the given index",
             VIEWER),
    Shortcut("toggle_bounding_boxes", ("d",), "Hold to hide the bounding boxes (Object List)", VIEWER,
             requires="napari_bbox"),
    Shortcut("run_script", ("Ctrl+Enter",), "Run the script (feature editor)", SCRIPT),
    Shortcut("script_font_larger", ("Ctrl-+",), "Increase the script editor font size", SCRIPT),
    Shortcut("script_font_smaller", ("Ctrl--",), "Decrease the script editor font size", SCRIPT),
)

BY_ID: Dict[str, Shortcut] = {s.id: s for s in SHORTCUTS}
assert len(BY_ID) == len(SHORTCUTS), "duplicate shortcut id"

Callback = Callable
# a callback, or for multi-key shortcuts a factory ``key -> callback``
Callbacks = Dict[str, Union[Callback, None]]


def _bind(target, shortcut: Shortcut, callback_for_key):
    for key in shortcut.keys:
        target.bind_key(key, overwrite=shortcut.overwrite)(callback_for_key(key))


def bind_all(target, scope: str, callbacks: Callbacks):
    """Bind the shortcuts named in ``callbacks`` (id -> callback) on a layer or viewer."""
    for sid, callback in callbacks.items():
        shortcut = BY_ID[sid]
        assert shortcut.scope == scope, "%s is a %r shortcut, not %r" % (sid, shortcut.scope, scope)
        multi = len(shortcut.keys) > 1
        _bind(target, shortcut, (lambda key, cb=callback: cb(key)) if multi else (lambda key, cb=callback: cb))


def unbind_all(target, scope: str, ids):
    """Remove the bindings of the shortcuts ``ids`` from a layer or viewer."""
    for sid in ids:
        shortcut = BY_ID[sid]
        assert shortcut.scope == scope, "%s is a %r shortcut, not %r" % (sid, shortcut.scope, scope)
        for key in shortcut.keys:
            target.bind_key(key, overwrite=True)(None)


def help_rows():
    """(action, shortcut) pairs of everything the plugin binds, in display order."""
    return [(s.description, s.display) for s in SHORTCUTS]


def markdown_table() -> str:
    """The README table: wheel variants in the Mouse column, key bindings in the Keyboard column."""
    lines = ["| Action | Mouse | Keyboard |", "|---|---|---|"]
    for s in SHORTCUTS:
        mouse = " / ".join(
            "`%s`%s" % (k.replace("↑", "⬆️").replace("↓", "⬇️"),
                        "\\*" if s.alt_builtin else "")
            for k in s.alt_keys) or "-"
        keyboard = "`%s`" % s.display_keys.replace("`", "")
        lines.append("| %s | %s | %s |" % (s.description, mouse, keyboard))
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown_table())
