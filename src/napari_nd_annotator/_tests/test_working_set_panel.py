"""The panel in which the labels layer, its image and its boxes are chosen and changed.

Like the other viewer tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools

import numpy as np
import pytest

pytest.importorskip("napari_bbox")

from .test_object_list import viewer, _process_events  # noqa: E402,F401
from napari_nd_annotator._widgets import working_set as ws  # noqa: E402
from napari_nd_annotator._widgets import working_set_panel as panel_module  # noqa: E402

SHAPE = (6, 24, 24)


def _select(viewer, layer):
    viewer.layers.selection.select_only(layer)
    _process_events()


@pytest.fixture
def scene(viewer, monkeypatch):
    """Two images of the same shape, a third of another shape, one labels layer with boxes, and an object list."""
    from napari_bbox import BoundingBoxLayer
    from napari_nd_annotator._widgets.object_list import ListWidgetBB
    warnings = []
    monkeypatch.setattr(panel_module, "show_warning", warnings.append)
    monkeypatch.setattr(ws, "show_info", lambda message: None)
    image = viewer.add_image(np.zeros(SHAPE), name="image")
    second = viewer.add_image(np.zeros(SHAPE), name="second image")
    flat = viewer.add_image(np.zeros((6, 24, 20)), name="other shape")
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels")
    boxes = BoundingBoxLayer(ndim=3, name="boxes", edge_color="green", face_color="transparent")
    viewer.add_layer(boxes)
    corners = np.where(list(itertools.product((False, True), repeat=3)), (4, 18, 18), (1, 4, 4))
    boxes.data = [corners.astype(float)]
    boxes.features = {"label": np.array([3])}
    ws.link_layers(image=image, labels=labels, boxes=boxes)
    viewer.layers.selection.clear()
    widget = ListWidgetBB(viewer)
    viewer.window.add_dock_widget(widget, name="Object list")
    _select(viewer, labels)
    return viewer, widget, widget.working_set_panel, (image, second, flat, labels, boxes), warnings


def _actions_of(menu):
    return {action.text(): action for action in menu.actions() if action.text()}


def test_the_panel_shows_the_layers_of_the_selected_labels_layer(scene):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    assert panel.labels_combobox.currentText() == "labels"
    assert panel.image_text.text().startswith("image") and panel.boxes_text.text() == "boxes"
    assert panel.image_button.isEnabled() and panel.boxes_button.isEnabled() and panel.new_boxes_button.isEnabled()


def test_choosing_a_labels_layer_selects_it_in_the_viewer(scene):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    other = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="other", metadata={"source_image_layer": second})
    _select(viewer, labels)
    panel.fill_labels_combobox()
    panel.labels_combobox.setCurrentIndex(panel.labels_combobox.findText("other"))
    panel._on_labels_chosen(panel.labels_combobox.currentIndex())
    _process_events()
    assert viewer.layers.selection.active is other
    assert panel.image_text.text().startswith("second image") and panel.boxes_text.text() == "none"
    assert widget.image.layer is second and widget.list_widget is None


def test_the_image_menu_lists_the_images_that_fit_and_says_why_not(scene):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    menu = _actions_of(panel.actions.image_menu(panel, ws.working_set(viewer)))
    assert menu["image (6x24x24)"].isChecked() and menu["image (6x24x24)"].isEnabled()
    assert menu["second image (6x24x24)"].isEnabled() and not menu["second image (6x24x24)"].isChecked()
    assert not menu["other shape (6x24x20)"].isEnabled()
    assert "different shapes" in menu["other shape (6x24x20)"].toolTip()


def test_changing_the_image_changes_the_link_and_the_list(scene):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    menu = _actions_of(panel.actions.image_menu(panel, ws.working_set(viewer)))
    menu["second image (6x24x24)"].trigger()
    _process_events()
    assert ws.linked_image(labels, viewer) is second and boxes.metadata["source_image_layer"] is second
    assert widget.image.layer is second and widget.list_widget.image_layer is second
    assert panel.image_text.text().startswith("second image")


def test_an_image_that_does_not_fit_is_refused_with_a_message(scene):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    panel.actions.change_image(labels, flat)
    assert ws.linked_image(labels, viewer) is image and len(warnings) == 1 and "different shapes" in warnings[0]


def test_the_boxes_menu_changes_and_removes_the_link(scene):
    from napari_bbox import BoundingBoxLayer
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    more = BoundingBoxLayer(ndim=3, name="more boxes", edge_color="green", face_color="transparent")
    viewer.add_layer(more)
    flat_boxes = BoundingBoxLayer(ndim=2, name="flat boxes", edge_color="green", face_color="transparent")
    viewer.add_layer(flat_boxes)
    _select(viewer, labels)
    menu = _actions_of(panel.actions.boxes_menu(panel, ws.working_set(viewer)))
    assert menu["boxes"].isChecked() and not menu["flat boxes"].isEnabled()
    menu["more boxes"].trigger()
    _process_events()
    assert ws.linked_boxes(labels, viewer) is more and widget.list_widget.bounding_box_layer is more
    _actions_of(panel.actions.boxes_menu(panel, ws.working_set(viewer)))["None"].trigger()
    _process_events()
    assert ws.linked_boxes(labels, viewer) is None
    assert widget.bounding_box.layer is None and widget.list_widget is None  # and it stays so
    assert panel.boxes_text.text() == "none"


def test_new_boxes_are_made_for_the_labels_layer_and_taken_by_the_list(scene):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    panel.actions.change_boxes(labels, None)
    _process_events()
    assert widget.list_widget is None
    panel.new_boxes_button.click()
    _process_events()
    new_boxes = ws.linked_boxes(labels, viewer)
    assert new_boxes is not None and new_boxes is not boxes and new_boxes in viewer.layers
    assert widget.bounding_box.layer is new_boxes and widget.list_widget is not None
    assert tuple(new_boxes.scale) == tuple(image.scale)


def test_new_labels_are_made_for_the_selected_image(scene):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    _select(viewer, second)
    assert panel.new_labels_button.isEnabled()
    panel.new_labels_button.click()
    _process_events()
    new = viewer.layers.selection.active
    assert new is not labels and type(new).__name__ == "Labels"
    assert new.data.shape == SHAPE and new.data.dtype == np.uint16 and not new.data.any()
    assert ws.linked_image(new, viewer) is second and ws.linked_boxes(new, viewer) is None
    assert widget.labels.layer is new and widget.image.layer is second


def test_no_image_no_new_labels(viewer):
    from napari_nd_annotator._widgets.working_set_panel import WorkingSetPanel
    panel = WorkingSetPanel(viewer)
    assert not panel.new_labels_button.isEnabled() and not panel.image_button.isEnabled()
    assert panel.labels_combobox.currentText() == "(no labels layer)"


def test_the_compact_panel_has_the_same_entries_in_one_menu(scene):
    from napari_nd_annotator._widgets.working_set_panel import WorkingSetPanel
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    compact = WorkingSetPanel(viewer, compact=True)
    compact.actions.fill_menu(compact.menu, ws.working_set(viewer))
    texts = [action.text() for action in compact.menu.actions() if action.text()]
    assert texts[:3] == ["Labels layer", "Image", "Boxes"] and any(text.startswith("New labels") for text in texts)
    assert "Image: <b>image</b>" in compact.status_text()


def _paint_two_objects(labels):
    labels.data[1:3, 2:6, 2:6] = 1
    labels.data[2:5, 10:20, 12:22] = 4
    labels.refresh()


def test_new_boxes_ask_for_boxes_of_the_painted_objects(scene, monkeypatch):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    _paint_two_objects(labels)
    panel.actions.change_boxes(labels, None)
    _process_events()
    questions = []
    monkeypatch.setattr(panel_module, "ask_yes_no", lambda title, text: questions.append(text) or True)
    panel.new_boxes_button.click()
    _process_events()
    new_boxes = ws.linked_boxes(labels, viewer)
    assert len(questions) == 1 and "2 objects" in questions[0] and "'labels'" in questions[0]
    assert list(new_boxes.features["label"]) == [1, 4]
    np.testing.assert_allclose(new_boxes.data[0].min(0), (1, 2, 2))
    np.testing.assert_allclose(new_boxes.data[0].max(0), (2, 5, 5))
    np.testing.assert_allclose(new_boxes.data[1].min(0), (2, 10, 12))
    np.testing.assert_allclose(new_boxes.data[1].max(0), (4, 19, 21))
    # they are in the list, and a box drawn next does not take an id that is painted
    assert widget.list_widget.count() == 2 and [widget.list_widget.item(i).idx for i in range(2)] == [1, 4]
    assert widget.list_widget.next_index() == 5


def test_new_boxes_can_be_made_without_boxes_for_the_painted_objects(scene, monkeypatch):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    _paint_two_objects(labels)
    panel.actions.change_boxes(labels, None)
    _process_events()
    monkeypatch.setattr(panel_module, "ask_yes_no", lambda title, text: False)
    panel.new_boxes_button.click()
    _process_events()
    new_boxes = ws.linked_boxes(labels, viewer)
    assert new_boxes is not None and len(new_boxes.data) == 0 and widget.list_widget.count() == 0
    # not even asked when nothing is painted
    labels.data[:] = 0
    questions = []
    monkeypatch.setattr(panel_module, "ask_yes_no", lambda title, text: questions.append(text) or True)
    panel.new_boxes_button.click()
    assert questions == []


def test_the_create_button_of_the_list_asks_as_well(scene, monkeypatch):
    viewer, widget, panel, (image, second, flat, labels, boxes), warnings = scene
    _paint_two_objects(labels)
    questions = []
    from napari_nd_annotator._widgets import object_list as object_list_module
    monkeypatch.setattr(panel_module, "ask_yes_no", lambda title, text: questions.append(text) or True)
    widget.create_bb_layer(False)
    _process_events()
    assert len(questions) == 1 and list(widget.bounding_box.layer.features["label"]) == [1, 4]
    assert widget.list_widget.count() == 2
