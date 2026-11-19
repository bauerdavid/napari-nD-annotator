"""The line that shows which image, labels and bounding box layers the tools work on (see `working_set`)."""
import html

from napari.layers import Labels
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QSizePolicy

from .._helper_functions import unwrap_viewer
from .working_set import (LAST_USED, SELECTED_OBJECT, WorkingSet, install, working_set, working_set_changed)

PROBLEM_COLOR = "#e05050"


def _shape(layer):
    try:
        return tuple(layer.data.shape[:layer.ndim])
    except (AttributeError, TypeError):
        return None


def problems(result: WorkingSet, viewer=None):
    """What is wrong with the layers of the working set (a list of sentences, empty if nothing)."""
    found = []
    image, labels = result.image, result.labels
    if labels is not None and image is None:
        found.append("The labels layer '%s' has no source image." % labels.name)
    if labels is not None and image is not None:
        image_shape, labels_shape = _shape(image), tuple(labels.data.shape)
        if image_shape is not None and image_shape != labels_shape:
            found.append("The labels layer '%s' %s and the image '%s' %s have different shapes."
                         % (labels.name, labels_shape, image.name, image_shape))
    active = viewer.layers.selection.active if viewer is not None else None
    if isinstance(active, Labels) and labels is not None and active is not labels:
        found.append("The active labels layer '%s' is not the one of the %s ('%s')."
                     % (active.name, "selected object" if result.source == SELECTED_OBJECT else "working set",
                        labels.name))
    return found


def source_description(result: WorkingSet):
    if result.source == SELECTED_OBJECT:
        return "from the selected object (label %d)" % result.object.label
    if result.source == LAST_USED:
        return "the layers used last (the active layer does not belong to a working set)"
    if result.source:
        return "from the active layer"
    return "no layers"


def status_html(result: WorkingSet, viewer=None):
    """The status line as html (names escaped), and its tooltip."""
    def name(layer):
        return html.escape(layer.name) if layer is not None else "<i>none</i>"

    text = "Image: <b>%s</b> &nbsp;|&nbsp; Labels: <b>%s</b> &nbsp;|&nbsp; Boxes: <b>%s</b>" % (
        name(result.image), name(result.labels), name(result.boxes))
    found = problems(result, viewer)
    for sentence in found:
        text += "<br><span style='color:%s'>%s</span>" % (PROBLEM_COLOR, html.escape(sentence))
    return text, "The layers the tools work on, %s." % source_description(result)


class WorkingSetStatus(QLabel):
    """Shows the working set of a viewer and follows it. Read only: the layers are chosen by selecting them."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self._viewer = unwrap_viewer(viewer)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        install(self._viewer)
        working_set_changed.connect(self._on_changed)
        # the active layer matters for the warnings, whether or not the working set changes with it
        self._viewer.layers.selection.events.active.connect(self._on_changed)
        self.refresh()

    def _on_changed(self, event=None):
        if getattr(event, "viewer", self._viewer) is not self._viewer:
            return
        try:
            self.refresh()
        except RuntimeError:  # the widget was deleted
            working_set_changed.disconnect(self._on_changed)
            self._viewer.layers.selection.events.active.disconnect(self._on_changed)

    def refresh(self):
        text, tooltip = status_html(working_set(self._viewer), self._viewer)
        self.setText(text)
        self.setToolTip(tooltip)
