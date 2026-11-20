"""The panel in which the layers the tools work on are shown and changed (see `working_set`).

The labels layer is the one selection: it is chosen in the combobox (which selects it in napari, like clicking it in
the layer list), and it knows its image and its box layer. Those two relations are changed with the ``Change...``
menus, and the missing layers are made with the ``New`` buttons. The compact form (the top of the Annotation Toolbox)
shows the status line and has one menu with the same entries.
"""
import html

from napari.layers import Image, Labels
from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QSizePolicy,
                            QToolButton, QVBoxLayout, QWidget)

from .._helper_functions import unwrap_viewer
from .working_set import (BoundingBoxLayer, LinkError, incompatibility, install, linked_boxes, linked_image,
                          new_boxes_for, new_labels_for, notify, objects_in_labels, set_boxes, set_source_image,
                          unlink_boxes, working_set, working_set_changed)
from .working_set_status import PROBLEM_COLOR, WorkingSetStatus, problems


def _warn(error):
    show_warning(str(error))


def ask_yes_no(title, text):
    """A question with Yes and No (the answer to a question that cannot be asked is No). Tests replace this."""
    return QMessageBox.question(None, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.Yes) == QMessageBox.StandardButton.Yes


def ask_boxes_for_painted_objects(labels):
    """Whether the new box layer of ``labels`` should get a box for each object that is painted in it already.

    Asked only if there are objects. Without boxes they are not in the object list, and the tools cannot work on them.
    """
    ids, _ = objects_in_labels(labels)
    if not ids:
        return False
    return ask_yes_no(
        "Boxes for the existing objects",
        "The labels layer '%s' has %d object%s already. Create a bounding box for %s?\n\n"
        "Objects without a box are not in the object list."
        % (labels.name, len(ids), "" if len(ids) == 1 else "s", "it" if len(ids) == 1 else "each"))


def _shape_text(layer):
    try:
        return "x".join(str(n) for n in layer.data.shape[:layer.ndim])
    except (AttributeError, TypeError):
        return ""


class WorkingSetActions:
    """What the panel does, with a viewer: the entries of the menus and the buttons."""

    def __init__(self, viewer):
        self.viewer = unwrap_viewer(viewer)

    def labels_layers(self):
        return [layer for layer in self.viewer.layers if isinstance(layer, Labels)]

    def select_labels(self, labels):
        self.viewer.layers.selection.select_only(labels)

    def change_image(self, labels, image):
        try:
            set_source_image(labels, image, self.viewer)
        except LinkError as error:
            _warn(error)
        notify(self.viewer)

    def change_boxes(self, labels, boxes):
        try:
            if boxes is None:
                unlink_boxes(labels)
            else:
                set_boxes(labels, boxes, self.viewer)
        except LinkError as error:
            _warn(error)
        notify(self.viewer)

    def new_labels(self, image):
        try:
            new_labels_for(self.viewer, image)
        except LinkError as error:
            _warn(error)
        notify(self.viewer)

    def new_boxes(self, labels):
        try:
            new_boxes_for(self.viewer, labels, from_objects=ask_boxes_for_painted_objects(labels))
        except LinkError as error:
            _warn(error)
        notify(self.viewer)

    def image_for_new_labels(self, result):
        """The image a new labels layer would be made for: the active one, else the one of the working set."""
        active = self.viewer.layers.selection.active
        if isinstance(active, Image) and not getattr(active, "multiscale", False):
            return active
        image = result.image
        return image if image is not None and not getattr(image, "multiscale", False) else None

    # --- the menus (built when they are shown, so they list the layers there are)

    def labels_menu(self, parent, result):
        menu = QMenu("Labels layer", parent)
        for labels in self.labels_layers():
            action = menu.addAction(labels.name)
            action.setCheckable(True)
            action.setChecked(labels is result.labels)
            action.triggered.connect(lambda _=False, layer=labels: self.select_labels(layer))
        if not menu.actions():
            menu.addAction("No labels layer").setEnabled(False)
        return menu

    def image_menu(self, parent, result):
        menu = QMenu("Image", parent)
        labels = result.labels
        images = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        for image in images:
            reason = incompatibility(image, labels) if labels is not None else "There is no labels layer."
            action = menu.addAction("%s (%s)" % (image.name, _shape_text(image)))
            action.setCheckable(True)
            action.setChecked(image is result.image)
            if reason is not None:
                action.setEnabled(False)
                action.setToolTip(reason)
            else:
                action.triggered.connect(lambda _=False, layer=image: self.change_image(labels, layer))
        if not images:
            menu.addAction("No image layer").setEnabled(False)
        return menu

    def boxes_menu(self, parent, result):
        menu = QMenu("Boxes", parent)
        labels = result.labels
        if BoundingBoxLayer is None:
            menu.addAction("Bounding boxes need the napari-bbox plugin").setEnabled(False)
            return menu
        for boxes in (layer for layer in self.viewer.layers if isinstance(layer, BoundingBoxLayer)):
            users = [other.name for other in self.labels_layers()
                     if other is not labels and linked_boxes(other, self.viewer) is boxes]
            action = menu.addAction(boxes.name + (" (also used by %s)" % ", ".join(users) if users else ""))
            action.setCheckable(True)
            action.setChecked(boxes is result.boxes)
            if labels is None or boxes.ndim != labels.ndim:
                action.setEnabled(False)
                action.setToolTip("The box layer needs the dimensions of the labels layer")
            else:
                action.triggered.connect(lambda _=False, layer=boxes: self.change_boxes(labels, layer))
        menu.addSeparator()
        none_action = menu.addAction("None")
        none_action.setEnabled(labels is not None and result.boxes is not None)
        none_action.triggered.connect(lambda _=False: self.change_boxes(labels, None))
        return menu

    def fill_menu(self, menu, result):
        """The whole menu of the compact panel."""
        for submenu in menu.findChildren(QMenu):
            submenu.deleteLater()
        menu.clear()
        menu.addMenu(self.labels_menu(menu, result))
        menu.addMenu(self.image_menu(menu, result))
        menu.addMenu(self.boxes_menu(menu, result))
        menu.addSeparator()
        image = self.image_for_new_labels(result)
        new_labels = menu.addAction("New labels for %s" % (image.name if image is not None else "the image"))
        new_labels.setEnabled(image is not None)
        new_labels.triggered.connect(lambda _=False: self.new_labels(image))
        new_boxes = menu.addAction("New boxes for the labels layer")
        new_boxes.setEnabled(result.labels is not None and result.image is not None and BoundingBoxLayer is not None)
        new_boxes.triggered.connect(lambda _=False: self.new_boxes(result.labels))


class _LabelsComboBox(QComboBox):
    """Lists the labels layers when it is opened."""

    def __init__(self, panel):
        super().__init__()
        self._panel = panel

    def showPopup(self):
        self._panel.fill_labels_combobox()
        super().showPopup()


class WorkingSetPanel(QWidget):
    """The layers the tools work on, shown and changed (see the module docstring)."""

    def __init__(self, viewer, compact=False, parent=None):
        super().__init__(parent)
        self._viewer = unwrap_viewer(viewer)
        self._compact = compact
        self.actions = WorkingSetActions(self._viewer)
        install(self._viewer)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if compact:
            self.status = WorkingSetStatus(self._viewer)
            self.menu_button = QToolButton()
            self.menu_button.setText("Layers...")
            self.menu_button.setToolTip("Change the layers the tools work on")
            self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            self.menu = QMenu(self.menu_button)
            self.menu.aboutToShow.connect(lambda: self.actions.fill_menu(self.menu, working_set(self._viewer)))
            self.menu_button.setMenu(self.menu)
            row = QHBoxLayout()
            row.addWidget(self.status, 1)
            row.addWidget(self.menu_button, 0, Qt.AlignmentFlag.AlignTop)
            layout.addLayout(row)
        else:
            self.status = None
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setColumnStretch(1, 1)
            self.labels_combobox = _LabelsComboBox(self)
            self.labels_combobox.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            self.labels_combobox.setToolTip("The labels layer the tools work on. Choosing one selects it in the viewer.")
            self.labels_combobox.activated.connect(self._on_labels_chosen)
            self.new_labels_button = QPushButton("New")
            self.new_labels_button.setToolTip("An empty labels layer for the selected image")
            self.new_labels_button.clicked.connect(self._on_new_labels)
            self.image_text = QLabel()
            self.image_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            self.image_button = QPushButton("Change...")
            self.image_button.setToolTip("The source image of the labels layer")
            self.image_button.clicked.connect(lambda: self._popup(self.image_button, self.actions.image_menu))
            self.boxes_text = QLabel()
            self.boxes_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            self.boxes_button = QPushButton("Change...")
            self.boxes_button.setToolTip("The box layer of the objects of the labels layer")
            self.boxes_button.clicked.connect(lambda: self._popup(self.boxes_button, self.actions.boxes_menu))
            self.new_boxes_button = QPushButton("New")
            self.new_boxes_button.setToolTip("A box layer for the objects of the labels layer")
            self.new_boxes_button.clicked.connect(self._on_new_boxes)
            for button in (self.new_labels_button, self.image_button, self.boxes_button, self.new_boxes_button):
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # the keyboard stays on the canvas
            grid.addWidget(QLabel("Labels"), 0, 0)
            grid.addWidget(self.labels_combobox, 0, 1)
            grid.addWidget(self.new_labels_button, 0, 2, 1, 2)
            grid.addWidget(QLabel("Image"), 1, 0)
            grid.addWidget(self.image_text, 1, 1)
            grid.addWidget(self.image_button, 1, 2, 1, 2)
            grid.addWidget(QLabel("Boxes"), 2, 0)
            grid.addWidget(self.boxes_text, 2, 1)
            grid.addWidget(self.boxes_button, 2, 2)
            grid.addWidget(self.new_boxes_button, 2, 3)
            layout.addLayout(grid)
            self.problem_label = QLabel()
            self.problem_label.setTextFormat(Qt.TextFormat.RichText)
            self.problem_label.setWordWrap(True)
            self.problem_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            layout.addWidget(self.problem_label)
            working_set_changed.connect(self._on_changed)
            self._viewer.layers.selection.events.active.connect(self._on_changed)
            self._viewer.layers.events.inserted.connect(self._on_changed)
            self._viewer.layers.events.removed.connect(self._on_changed)
            self.refresh()

    # --- the full form

    def _on_changed(self, event=None):
        if getattr(event, "viewer", self._viewer) is not self._viewer:
            return
        try:
            self.refresh()
        except RuntimeError:  # the widget was deleted
            working_set_changed.disconnect(self._on_changed)
            for emitter in (self._viewer.layers.selection.events.active, self._viewer.layers.events.inserted,
                            self._viewer.layers.events.removed):
                emitter.disconnect(self._on_changed)

    def fill_labels_combobox(self):
        result = working_set(self._viewer)
        self.labels_combobox.blockSignals(True)
        self.labels_combobox.clear()
        layers = self.actions.labels_layers()
        for labels in layers:
            self.labels_combobox.addItem(labels.name)
        if result.labels in layers:
            self.labels_combobox.setCurrentIndex(layers.index(result.labels))
        elif not layers:
            self.labels_combobox.addItem("(no labels layer)")
        self.labels_combobox.setEnabled(bool(layers))
        self.labels_combobox.blockSignals(False)

    def refresh(self):
        if self._compact:
            return
        result = working_set(self._viewer)
        self.fill_labels_combobox()
        self.image_text.setText("%s  %s" % (result.image.name, _shape_text(result.image)) if result.image is not None
                                else "none")
        self.boxes_text.setText(result.boxes.name if result.boxes is not None else "none")
        has_labels = result.labels is not None
        self.image_button.setEnabled(has_labels)
        self.boxes_button.setEnabled(has_labels and BoundingBoxLayer is not None)
        self.new_boxes_button.setEnabled(has_labels and result.image is not None and BoundingBoxLayer is not None)
        self.new_labels_button.setEnabled(self.actions.image_for_new_labels(result) is not None)
        found = problems(result, self._viewer)
        self.problem_label.setText("<br>".join("<span style='color:%s'>%s</span>" % (PROBLEM_COLOR, html.escape(text))
                                               for text in found))
        self.problem_label.setVisible(bool(found))
        self.setToolTip("The layers the tools work on")

    def _on_labels_chosen(self, index):
        layers = self.actions.labels_layers()
        if 0 <= index < len(layers):
            self.actions.select_labels(layers[index])

    def _popup(self, button, build_menu):
        menu = build_menu(self, working_set(self._viewer))
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def _on_new_labels(self):
        image = self.actions.image_for_new_labels(working_set(self._viewer))
        if image is not None:
            self.actions.new_labels(image)

    def _on_new_boxes(self):
        labels = working_set(self._viewer).labels
        if labels is not None:
            self.actions.new_boxes(labels)

    def status_text(self):
        """The text of the layers shown (for tests and tooltips)."""
        if self._compact:
            return self.status.text()
        return "Labels: %s | Image: %s | Boxes: %s" % (
            self.labels_combobox.currentText(), self.image_text.text(), self.boxes_text.text())
