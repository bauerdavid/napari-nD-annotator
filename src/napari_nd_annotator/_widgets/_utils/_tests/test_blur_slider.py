from ..blur_slider import BlurSlider
import numpy as np
from skimage.filters import median
from skimage.morphology import disk


def test_blur_slider_add_image_later(make_napari_viewer):
    viewer = make_napari_viewer()
    blur_slider = BlurSlider(viewer)
    blur_slider.setValue(11)
    image_layer = viewer.add_image(np.random.rand(100, 100))
    blur_slider.image_layer = image_layer
    assert np.allclose(blur_slider.blur_func(image_layer.data, blur_slider.value()), blur_slider.get_blurred_image())


def test_blur_slider_w_image(make_napari_viewer):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(np.random.rand(100, 100))
    blur_slider = BlurSlider(viewer, image_layer)
    blur_slider.setValue(11)
    assert np.allclose(blur_slider.blur_func(image_layer.data, blur_slider.value()), blur_slider.get_blurred_image())


def test_blur_slider_custom_blur(make_napari_viewer):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(np.random.rand(100, 100))
    blur_func = lambda img, val: median(img, disk(val))
    blur_slider = BlurSlider(viewer, image_layer, blur_func)
    blur_slider.setValue(11)
    assert np.allclose(blur_slider.blur_func(image_layer.data, blur_slider.value()), blur_slider.get_blurred_image())
