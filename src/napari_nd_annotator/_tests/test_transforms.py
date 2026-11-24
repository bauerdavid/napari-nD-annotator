"""The transform helpers: layers that show (part of) the voxels of another one must show them in the same place."""
import numpy as np
from napari.layers import Image, Labels

from napari_nd_annotator._helper_functions import copy_transform, layer_transform


def _image(**kwargs):
    return Image(np.zeros((10, 20, 30)), **kwargs)


TRANSFORM = dict(scale=(3., .5, 2.), translate=(1., -2., 5.), rotate=35., shear=[.1, 0., .2],
                 affine=np.array([[1., 0., 0., 4.], [0., 1., .3, 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]))


def test_a_crop_resampled_along_z_shows_its_voxels_where_the_original_has_them():
    image = _image(**TRANSFORM)
    offset, factor = np.array([2., 5., 7.]), np.array([1 / 3, 1., 1.])
    crop = Image(np.zeros((9, 4, 4)), **layer_transform(image, offset=offset, factor=factor))
    for index in ((0, 0, 0), (1, 2, 3), (8, 3, 3), (4.5, .5, 1.)):
        np.testing.assert_allclose(crop.data_to_world(index), image.data_to_world(offset + factor * np.asarray(index)))


def test_copy_transform_gives_the_same_world_positions_and_says_what_changed():
    image = _image(**TRANSFORM)
    labels = Labels(np.zeros((10, 20, 30), dtype=np.uint8), scale=(1., .5, 2.))
    changed = copy_transform(image, labels)
    assert set(changed) == {"scale", "translate", "rotate", "shear", "affine"}
    for index in ((0, 0, 0), (3, 7, 11)):
        np.testing.assert_allclose(labels.data_to_world(index), image.data_to_world(index))
    assert copy_transform(image, labels) == []


def test_copy_transform_between_layers_of_different_dimensionality_uses_the_last_dimensions():
    channels = Image(np.zeros((2, 10, 20, 30)), scale=(1., 3., .5, 2.), translate=(0., 1., -2., 5.))
    labels = Labels(np.zeros((10, 20, 30), dtype=np.uint8))
    copy_transform(channels, labels)
    np.testing.assert_allclose(labels.scale, (3., .5, 2.))
    np.testing.assert_allclose(labels.translate, (1., -2., 5.))
