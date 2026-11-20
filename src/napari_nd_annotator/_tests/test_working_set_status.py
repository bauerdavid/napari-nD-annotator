"""The line that shows the working set, and the tools that stay on their labels layer when a helper layer is active.

Like the other viewer tests these need a GL-capable Qt platform, and the napari-bbox plugin.
"""
import itertools

import numpy as np
import pytest

pytest.importorskip("napari_bbox")

from .test_object_list import viewer, _process_events  # noqa: E402,F401
from napari_nd_annotator._widgets import working_set as ws  # noqa: E402
from napari_nd_annotator._widgets.working_set_status import problems, status_html  # noqa: E402

SHAPE = (6, 20, 20)


def _select(viewer, layer):
    viewer.layers.selection.select_only(layer)
    _process_events()


@pytest.fixture
def annotator(viewer):
    from napari_nd_annotator import AnnotatorWidget
    widget = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(widget, name="Annotator")
    return widget


@pytest.fixture
def scene(viewer):
    image = viewer.add_image(np.zeros(SHAPE), name="image")
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels", metadata={"source_image_layer": image})
    return image, labels


def test_the_status_names_the_layers_and_escapes_them(viewer):
    image = viewer.add_image(np.zeros(SHAPE), name="a <b>")
    text, tooltip = status_html(ws.WorkingSet(image=image, source=ws.ACTIVE_LAYER), viewer)
    assert "a &lt;b&gt;" in text and "<b>a <b>" not in text
    assert "Labels: <b><i>none</i></b>" in text and "active layer" in tooltip


def test_mismatching_shapes_and_missing_sources_are_reported(viewer):
    image = viewer.add_image(np.zeros(SHAPE), name="image")
    labels = viewer.add_labels(np.zeros((6, 20, 30), dtype=np.uint16), name="labels")
    assert problems(ws.WorkingSet(image=image, labels=labels), viewer)[0].endswith("different shapes.")
    assert "no source image" in problems(ws.WorkingSet(labels=labels), viewer)[0]
    assert problems(ws.WorkingSet(image=image), viewer) == []


def test_another_active_labels_layer_than_the_one_of_the_set_is_reported(viewer, scene):
    image, labels = scene
    other = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="other", metadata={"source_image_layer": image})
    _select(viewer, other)
    found = problems(ws.WorkingSet(image=image, labels=labels, source=ws.ACTIVE_LAYER), viewer)
    assert len(found) == 1 and "'other'" in found[0]
    _select(viewer, labels)
    assert problems(ws.WorkingSet(image=image, labels=labels, source=ws.ACTIVE_LAYER), viewer) == []


def test_the_line_in_the_toolbox_and_the_panel_in_the_object_list_follow_the_selection(viewer, annotator, scene):
    from napari_nd_annotator._widgets.object_list import ListWidgetBB
    image, labels = scene
    object_list = ListWidgetBB(viewer)
    viewer.window.add_dock_widget(object_list, name="Object list")
    _select(viewer, labels)
    assert "Image: <b>image</b>" in annotator.working_set_panel.status_text()
    assert "Labels: <b>labels</b>" in annotator.working_set_panel.status_text()
    panel = object_list.working_set_panel
    assert panel.labels_combobox.currentText() == "labels" and panel.image_text.text().startswith("image")
    before = annotator.working_set_panel.status_text()
    _select(viewer, viewer.add_points(np.zeros((1, 3)), name="points"))
    assert annotator.working_set_panel.status_text() == before  # a helper layer changes nothing
    other = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="other", metadata={"source_image_layer": image})
    _select(viewer, other)
    assert "Labels: <b>other</b>" in annotator.working_set_panel.status_text()
    assert panel.labels_combobox.currentText() == "other"


def test_a_helper_layer_does_not_switch_the_tools_off(viewer, annotator, scene):
    image, labels = scene
    minimal_contour = annotator.ToolsWidget.minimal_contour_widget
    interpolation = annotator.ToolsWidget.interpolation_widget
    _select(viewer, labels)
    assert minimal_contour.labels_layer is labels and interpolation.active_labels_layer is labels
    _select(viewer, viewer.add_points(np.zeros((1, 3)), name="points"))
    assert minimal_contour.labels_layer is labels and interpolation.active_labels_layer is labels
    assert not minimal_contour.non_labels_layer_txt.visible
    assert interpolation.interpolate_button.enabled


def test_the_layer_of_the_bounding_boxes_does_not_switch_the_tools_off(viewer, annotator, scene):
    from napari_bbox import BoundingBoxLayer
    image, labels = scene
    boxes = BoundingBoxLayer(ndim=3, edge_color="green", face_color="transparent")
    viewer.add_layer(boxes)
    ws.link_layers(image=image, labels=labels, boxes=boxes)
    minimal_contour = annotator.ToolsWidget.minimal_contour_widget
    _select(viewer, labels)
    _select(viewer, boxes)  # to draw a box
    assert minimal_contour.labels_layer is labels
    assert minimal_contour.image_layer is image


def test_the_tools_are_off_when_there_is_nothing_to_work_on(viewer, annotator):
    _select(viewer, viewer.add_points(np.zeros((1, 3)), name="points"))
    assert annotator.ToolsWidget.minimal_contour_widget.labels_layer is None
    assert annotator.ToolsWidget.interpolation_widget.active_labels_layer is None


def test_a_labels_layer_without_a_source_does_not_ask_for_one_from_a_helper_layer(viewer, annotator, scene, monkeypatch):
    """Only selecting the labels layer itself may ask for its source image (a dialog)."""
    image, labels = scene
    del labels.metadata["source_image_layer"]
    asked = []
    minimal_contour = annotator.ToolsWidget.minimal_contour_widget
    # (patched on the class: magicgui containers do not take the removal of an attribute well)
    monkeypatch.setattr(type(minimal_contour), "_select_source_image_dialog",
                        lambda self, layer: asked.append(layer) or False)
    _select(viewer, viewer.add_points(np.zeros((1, 3)), name="points"))
    assert asked == [] and minimal_contour.labels_layer is None


def test_boxes_of_another_image_are_reported(viewer):
    from napari_bbox import BoundingBoxLayer
    image = viewer.add_image(np.zeros(SHAPE), name="image")
    other = viewer.add_image(np.zeros(SHAPE), name="other")
    labels = viewer.add_labels(np.zeros(SHAPE, dtype=np.uint16), name="labels")
    boxes = BoundingBoxLayer(ndim=3, name="boxes", edge_color="green", face_color="transparent")
    viewer.add_layer(boxes)
    ws.link_layers(image=other, boxes=boxes)
    found = problems(ws.WorkingSet(image=image, labels=labels, boxes=boxes), viewer)
    assert len(found) == 1 and "'other'" in found[0] and "'image'" in found[0]
