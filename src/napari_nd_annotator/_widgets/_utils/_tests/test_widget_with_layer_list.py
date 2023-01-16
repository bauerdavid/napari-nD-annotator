from napari.layers import Image
import numpy as np
from ..widget_with_layer_list import WidgetWithLayerList
import unittest
import pytest


class TestWidgetWithLayerList(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_make_napari_viewer(self, make_napari_viewer):
        self.make_napari_viewer = make_napari_viewer

    def test_list_updated_on_layer_added(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        viewer.add_image(np.zeros((100, 100)), name="MyImage")
        self.assertEqual(wwll.image.layer.name, "MyImage")
    
    
    def test_list_updated_on_layer_deleted(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        viewer.add_image(np.zeros((100, 100)), name="MyImage")
        viewer.layers.remove("MyImage")
        self.assertIsNone(wwll.image.layer)
    
    
    def test_list_updated_on_layer_move(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        viewer.add_image(np.zeros((100, 100)), name="image1")
        viewer.add_image(np.zeros((100, 100)), name="image2")
        self.assertEqual(viewer.layers.index("image1"), wwll.image.combobox.findText("image1")-1)
        self.assertEqual(viewer.layers.index("image2"), wwll.image.combobox.findText("image2")-1)
        viewer.layers.move(0, 1)
        self.assertEqual(viewer.layers.index("image1"), wwll.image.combobox.findText("image1")-1)
        self.assertEqual(viewer.layers.index("image2"), wwll.image.combobox.findText("image2")-1)
    
    
    def test_list_updated_on_layer_renamed(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        viewer.add_image(np.zeros((100, 100)), name="MyImage")
        viewer.layers["MyImage"].name = "MyImageRenamed"
        self.assertGreater(wwll.image.combobox.findText("MyImageRenamed"), -1)
    
    
    def test_layer_selection_by_layer(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        image1 = viewer.add_image(np.zeros((100, 100)), name="image1")
        image2 = viewer.add_image(np.zeros((100, 100)), name="image2")
        self.assertEqual(wwll.image.combobox.currentText(), "image1")
        wwll.image.layer = image2
        self.assertEqual(wwll.image.combobox.currentText(), "image2")
    
    
    def test_layer_selection_by_name(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        image1 = viewer.add_image(np.zeros((100, 100)), name="image1")
        image2 = viewer.add_image(np.zeros((100, 100)), name="image2")
        self.assertEqual(wwll.image.combobox.currentText(), "image1")
        wwll.image.layer = "image2"
        self.assertEqual(wwll.image.combobox.currentText(), "image2")
    
    
    def test_layer_selection_error_on_wrong_type(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        image = viewer.add_image(np.zeros((100, 100)), name="image")
        labels = viewer.add_labels(np.zeros((100, 100), dtype=int), name="labels")

        def callback():
            wwll.image.layer = labels
        self.assertRaises(TypeError, callback)
    
    
    def test_call_on_layer_selection(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        val = [None]
    
        def callback(idx):
            val[0] = idx
    
        wwll.image.combobox.currentIndexChanged.connect(callback)
        image1 = viewer.add_image(np.zeros((100, 100)), name="image1")
        self.assertEqual(val[0] - 1, viewer.layers.index("image1"))
        image2 = viewer.add_image(np.zeros((100, 100)), name="image2")
        wwll.image.layer = image2
        self.assertEqual(val[0] - 1, viewer.layers.index("image2"))
    
    
    def test_call_on_layer_selection_only_once(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        val = [0]
    
        def callback(_):
            val[0] += 1
    
        wwll.image.combobox.currentIndexChanged.connect(callback)
        image1 = viewer.add_image(np.zeros((100, 100)), name="image1")
        self.assertEqual(val[0], 1)
    
    
    def test_layer_is_none_when_list_empty(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        self.assertIsNone(wwll.image.layer, None)
    
    
    def test_layer_is_selected_if_layerlist_not_empty(self):
        viewer = self.make_napari_viewer()
        wwll = WidgetWithLayerList(viewer, [("image", Image)])
        image1 = viewer.add_image(np.zeros((100, 100)), name="image1")
        wwll.image.combobox.setCurrentIndex(0)
        self.assertGreater(wwll.image.combobox.currentIndex(), 0)