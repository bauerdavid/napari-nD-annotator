"""Selected-label bounding box: one overlay per Labels layer, switched from the layer controls.

``install_selected_label_bbox(viewer)`` adds a :class:`SelectedLabelBBoxOverlay`
to every Labels layer of the viewer (now and later) and a "selected bbox:" row
to the layer's controls with a checkbox (show/hide) and a color swatch.
"""
import weakref

import numpy as np
from napari.layers import Labels
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from .selected_label_bbox_overlay import SelectedLabelBBoxOverlay
from ..._helper_functions import unwrap_viewer, viewer_qt_viewer

OVERLAY_KEY = "selected_label_bbox"
CONTROLS_ROW_LABEL = "selected bbox:"

_installed_viewers = weakref.WeakSet()


def get_bbox_overlay(layer: Labels) -> SelectedLabelBBoxOverlay:
    """The layer's selected-label box overlay, added on first use."""
    overlay = layer._overlays.get(OVERLAY_KEY)
    if overlay is None:
        overlay = SelectedLabelBBoxOverlay()
        layer._overlays.update({OVERLAY_KEY: overlay})
    return overlay


class QtSelectedLabelBBoxControls(QWidget):
    """Checkbox and color swatch bound to a layer's selected-label box overlay."""

    def __init__(self, layer: Labels, parent=None):
        super().__init__(parent)
        # imported here: napari's private Qt widgets are only needed once a viewer exists
        from napari._qt.widgets.qt_color_swatch import QColorSwatch

        self._overlay = get_bbox_overlay(layer)

        self.checkbox = QCheckBox()
        self.checkbox.setToolTip("Show a box around the selected label")
        self.checkbox.setChecked(self._overlay.visible)
        self.checkbox.toggled.connect(self._on_checkbox_toggled)

        self.color_swatch = QColorSwatch(
            initial_color=np.asarray(self._overlay.color),
            tooltip="Click to set the color of the selected label's box",
        )
        self.color_swatch.color_changed.connect(self._on_swatch_changed)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.color_swatch, 1)
        self.setLayout(layout)

        overlay = self._overlay
        on_visible, on_color = self._on_overlay_visible_change, self._on_overlay_color_change
        overlay.events.visible.connect(on_visible)
        overlay.events.color.connect(on_color)

        def disconnect_overlay(*_):
            overlay.events.visible.disconnect(on_visible)
            overlay.events.color.disconnect(on_color)
        # the layer controls (our parent) are deleted with the layer
        self.destroyed.connect(disconnect_overlay)

    def _on_checkbox_toggled(self, checked):
        self._overlay.visible = checked

    def _on_swatch_changed(self, color):
        if not np.array_equal(np.asarray(self._overlay.color), color):
            self._overlay.color = color

    def _on_overlay_visible_change(self, *_):
        # setChecked only emits (and calls back into the overlay) when the state differs
        self.checkbox.setChecked(self._overlay.visible)

    def _on_overlay_color_change(self, *_):
        # not blocked: the swatch repaints itself from its own color_changed signal;
        # _on_swatch_changed ignores the echo
        self.color_swatch.setColor(np.asarray(self._overlay.color))


def add_bbox_controls(viewer, layer: Labels) -> bool:
    """Add the overlay to ``layer`` and its row to the layer controls. Returns False if the controls don't exist yet."""
    if not isinstance(layer, Labels):
        return False
    get_bbox_overlay(layer)
    controls = viewer_qt_viewer(viewer).controls.widgets.get(layer)
    if controls is None:
        return False
    if getattr(controls, "selected_label_bbox_controls", None) is not None:
        return True
    bbox_controls = QtSelectedLabelBBoxControls(layer, parent=controls)
    controls.selected_label_bbox_controls = bbox_controls
    controls.layout().addRow(CONTROLS_ROW_LABEL, bbox_controls)
    return True


def install_selected_label_bbox(viewer):
    """Give every current and future Labels layer of ``viewer`` the selected-label box and its controls.

    Safe to call more than once per viewer.
    """
    viewer = unwrap_viewer(viewer)
    if viewer in _installed_viewers:
        return
    _installed_viewers.add(viewer)

    def on_inserted(event):
        layer = event.value
        if isinstance(layer, Labels) and not add_bbox_controls(viewer, layer):
            # the layer controls may be created after us; try again once the event has been handled
            QTimer.singleShot(0, lambda: layer in viewer.layers and add_bbox_controls(viewer, layer))

    viewer.layers.events.inserted.connect(on_inserted)
    for layer in viewer.layers:
        add_bbox_controls(viewer, layer)
