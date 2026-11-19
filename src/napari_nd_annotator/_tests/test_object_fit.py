"""Fitting a bounding box to the object under it (grows along the dimension the box does not span)."""
import numpy as np
import pytest

from napari_nd_annotator.object_fit import fit_box_to_object

SCALE = np.array([3., 1., 1.])
SHAPE = (40, 120, 120)
CENTER = np.array([20., 50., 50.])      # in voxels
RADII = np.array([10., 30., 30.])       # in voxels: a ball of radius 30 in physical units
# what a perfect fit finds
TRUE_MIN = (CENTER - RADII).astype(int)
TRUE_MAX = (CENTER + RADII).astype(int)


def _ball(center, radii):
    grid = np.mgrid[tuple(slice(0, s) for s in SHAPE)].astype(float)
    return (sum(((g - c) / r) ** 2 for g, c, r in zip(grid, center, radii)) <= 1).astype(float)


@pytest.fixture
def image():
    """A bright ball of radius 30 (physical units) and another one nearby, with noise."""
    data = _ball(CENTER, RADII) + _ball(np.array([20., 100., 100.]), np.array([6., 15., 15.]))
    return data * 100 + 20 + np.random.default_rng(0).normal(0, 5, SHAPE)


def _check(fit, tolerance=2):
    assert fit.found, fit.reason
    assert np.all(np.abs(fit.box_min - TRUE_MIN) <= tolerance), (fit.box_min, TRUE_MIN)
    assert np.all(np.abs(fit.box_max - TRUE_MAX) <= tolerance), (fit.box_max, TRUE_MAX)


def test_a_flat_box_in_the_middle_grows_along_the_hidden_dimension(image):
    # drawn on the middle slice, too small and off the middle
    fit = fit_box_to_object(image, (20, 40, 45), (20, 60, 70), scale=SCALE)
    _check(fit)
    assert fit.polarity == "bright" and not fit.reaches_limit


def test_a_flat_box_drawn_from_the_top_grows_in_every_dimension(image):
    # on the first slices of the object, where its cross section is a small disc: the size in the plane is not known
    top = int(TRUE_MIN[0]) + 1
    fit = fit_box_to_object(image, (top, 42, 42), (top, 58, 58), scale=SCALE)
    _check(fit, tolerance=3)


def test_a_box_that_is_too_large_shrinks_to_the_object(image):
    fit = fit_box_to_object(image, (5, 20, 20), (35, 80, 85), scale=SCALE)
    _check(fit)


def test_dark_objects(image):
    fit = fit_box_to_object(-image, (20, 40, 45), (20, 60, 70), scale=SCALE)
    _check(fit)
    assert fit.polarity == "dark"


def test_the_polarity_can_be_given(image):
    assert fit_box_to_object(image, (20, 40, 45), (20, 60, 70), scale=SCALE, polarity="dark").found is False
    _check(fit_box_to_object(image, (20, 40, 45), (20, 60, 70), scale=SCALE, polarity="bright"))
    with pytest.raises(ValueError):
        fit_box_to_object(image, (20, 40, 45), (20, 60, 70), polarity="gray")


def test_the_corner_order_does_not_matter(image):
    fit = fit_box_to_object(image, (20, 60, 70), (20, 40, 45), scale=SCALE)
    _check(fit)


def test_a_box_on_the_background_finds_nothing(image):
    fit = fit_box_to_object(image, (20, 100, 5), (20, 110, 15), scale=SCALE)
    assert not fit.found and fit.reason


def test_a_box_outside_of_the_image_finds_nothing(image):
    assert not fit_box_to_object(image, (20, 300, 300), (20, 310, 310), scale=SCALE).found


def test_an_object_larger_than_the_search_region_is_reported(image):
    fit = fit_box_to_object(image, (20, 40, 40), (20, 60, 60), scale=SCALE, search_factor=2.)
    assert fit.found and not fit.reaches_limit
    # a tiny box deep inside the object has no background in its search region: the region is widened
    tiny = fit_box_to_object(image, (20, 48, 48), (20, 52, 52), scale=SCALE, search_factor=1.)
    _check(tiny)
    # an object that is larger than even the widest search region (8 times the box): flagged
    big = np.full((10, 40, 400), 20.)
    big[:, :, 30:] += 100
    big += np.random.default_rng(2).normal(0, 2, big.shape)
    fit = fit_box_to_object(big, (5, 20, 35), (5, 25, 45), search_factor=1.)
    assert fit.found and fit.reaches_limit


def test_without_a_scale_the_voxels_are_isotropic():
    data = np.zeros((30, 30, 30)) + 10.
    data[10:20, 8:22, 5:15] = 200.
    fit = fit_box_to_object(data + np.random.default_rng(1).normal(0, 2, data.shape), (15, 10, 8), (15, 18, 12))
    assert fit.found
    np.testing.assert_allclose(fit.box_min, (10, 8, 5), atol=1)
    np.testing.assert_allclose(fit.box_max, (19, 21, 14), atol=1)
