"""The 'autofill objects' option of the Annotation Toolbox is on by default, a saved choice still wins."""
import json

import pytest


@pytest.fixture
def viewer():
    import napari
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()


def test_autofill_objects_is_on_by_default(viewer):
    from napari_nd_annotator import AnnotatorWidget
    annotator = AnnotatorWidget(viewer, persist=False)
    assert annotator.autofill_objects is True


def test_a_saved_choice_is_not_overridden(viewer, tmp_path, monkeypatch):
    from magicclass.serialize import serialize
    from napari_nd_annotator import AnnotatorWidget
    dump_path = tmp_path / "annotator_dump"
    monkeypatch.setattr(AnnotatorWidget, "_dump_path", property(lambda self: dump_path))
    first = AnnotatorWidget(viewer, persist=False)
    first.autofill_objects = False
    dump_path.write_text(json.dumps(serialize(first)))
    second = AnnotatorWidget(viewer, persist=True)
    assert second.autofill_objects is False
