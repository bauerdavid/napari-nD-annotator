"""The anchor store and the ray/plane picking of ``object_anchors`` (no viewer needed)."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from napari_nd_annotator.object_anchors import (
    AnchorStore, METADATA_KEY, anchor_store, clip_ray_to_box, pick_along_ray, pick_on_plane)

CENTER = np.array([20., 30., 30.])
RADIUS = 12.
SHAPE = (40, 60, 60)
BOX = (CENTER - 15, CENTER + 15)


def _ball(noise=0.):
    zz, yy, xx = np.mgrid[tuple(slice(0, s) for s in SHAPE)]
    distance = np.sqrt(((np.stack([zz, yy, xx], -1) - CENTER) ** 2).sum(-1))
    image = gaussian_filter((distance <= RADIUS).astype(float), 1) * 1000
    return image + np.random.default_rng(0).normal(0, noise, SHAPE)


def test_clip_ray_to_box():
    assert clip_ray_to_box([0, 0, 0], [10, 0, 0], [2, 0, 0], [4, 0, 0]) == pytest.approx((1.5 / 10, 4.5 / 10))
    # a ray along an axis outside the box in another axis misses it
    assert clip_ray_to_box([0, 5, 0], [10, 5, 0], [2, 0, 0], [4, 1, 1]) is None
    # a segment that ends before the box
    assert clip_ray_to_box([0, 0, 0], [1, 0, 0], [5, 0, 0], [8, 0, 0]) is None
    # starting inside the box
    assert clip_ray_to_box([3, 0, 0], [10, 0, 0], [2, 0, 0], [4, 0, 0])[0] == 0


@pytest.mark.parametrize("direction", [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1), (-1, 2, 1), (0, -1, 0)])
def test_ray_picks_the_surface_of_a_ball(direction):
    image = _ball(noise=20)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    # the ray comes from far outside the ball, towards its center
    start = CENTER - 40 * direction
    point = pick_along_ray(image, start, CENTER + 40 * direction, *BOX)
    assert point is not None
    assert abs(np.linalg.norm(point - CENTER) - RADIUS) < 1.5
    # on the side the ray comes from
    assert (point - CENTER) @ direction < 0


def test_ray_ignores_what_is_outside_the_box():
    image = _ball()
    # a second ball next to the box would be hit first by the ray, but it is outside
    image[:, :, :8] = 1000.
    start = np.array([20., 30., -20.])
    point = pick_along_ray(image, start, start + [0, 0, 100.], CENTER - 15, CENTER + 15)
    assert abs(point[2] - (CENTER[2] - RADIUS)) < 1.5


def test_ray_picks_the_first_structure_of_two():
    image = _ball()
    image[:, :, 48:56] = 1000.  # a wall behind the ball
    start = np.array([20., 30., 0.])
    point = pick_along_ray(image, start, start + [0, 0, 59.], np.array([5, 15, 15]), np.array([35, 45, 55]))
    assert abs(point[2] - (CENTER[2] - RADIUS)) < 1.5


def test_ray_that_misses_the_data_or_the_box_gives_none():
    image = _ball(noise=20)
    # through the box, but beside the ball
    start = np.array([20., 8., 0.])
    assert pick_along_ray(image, start, start + [0, 0, 59.], *BOX) is None
    # not through the box at all
    start = np.array([39., 59., 0.])
    assert pick_along_ray(image, start, start + [0, 0, 59.], *BOX) is None
    # a flat volume has no contrast
    assert pick_along_ray(np.zeros(SHAPE), CENTER - 30, CENTER + 30, *BOX) is None


def test_ray_starting_inside_the_structure_picks_the_box_wall():
    image = _ball()
    point = pick_along_ray(image, CENTER, CENTER + [0, 0, 40.], *BOX)
    np.testing.assert_allclose(point, CENTER)


def test_ray_works_on_integer_images_and_does_not_modify_them():
    image = _ball(noise=20).clip(0, None).astype(np.uint16)
    before = image.copy()
    point = pick_along_ray(image, CENTER - [0, 0, 40], CENTER + [0, 0, 40], *BOX)
    assert abs(np.linalg.norm(point - CENTER) - RADIUS) < 1.5
    np.testing.assert_array_equal(image, before)


def test_plane_pick():
    point = pick_on_plane([0, 0, 0], [10, 10, 10], plane_position=[5, 5, 5], plane_normal=[0, 0, 1],
                          box_min=[0, 0, 0], box_max=[9, 9, 9])
    np.testing.assert_allclose(point, [5, 5, 5])
    # oblique plane
    point = pick_on_plane([0, 0, 0], [1, 0, 0], [3, 0, 0], [1, 1, 0], [0, -5, -5], [9, 5, 5])
    np.testing.assert_allclose(point, [3, 0, 0])
    # parallel, and crossing outside the box
    assert pick_on_plane([0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 1, 0], [0, 0, 0], [9, 9, 9]) is None
    assert pick_on_plane([0, 0, 0], [0, 0, 1], [0, 0, 20], [0, 0, 1], [0, 0, 0], [9, 9, 9]) is None


def test_store_pairs():
    store = AnchorStore()
    assert len(store.get(3)) == 0 and not store.has_pair(3)
    store.add(3, [1, 2, 3])
    assert store.get(3).shape == (1, 3) and not store.has_pair(3)
    store.add(3, [4, 5, 6])
    assert store.has_pair(3)
    np.testing.assert_array_equal(store.get(3), [[1, 2, 3], [4, 5, 6]])
    # a third point starts a new pair
    store.add(3, [7, 8, 9])
    np.testing.assert_array_equal(store.get(3), [[7, 8, 9]])
    store.add(4, [0, 0, 0])
    store.pop_last(3)
    assert len(store.get(3)) == 0 and len(store.get(4)) == 1
    store.clear(4)
    assert len(store.get(4)) == 0
    with pytest.raises(ValueError):
        store.set(1, [[0, 0, 0]] * 3)
    # json friendly
    store.set(5, np.array([[1., 2., 3.], [4., 5., 6.]]))
    import json
    assert json.loads(json.dumps(store.storage)) == {"5": [[1., 2., 3.], [4., 5., 6.]]}


def test_store_of_a_layer_lives_in_its_metadata():
    class Layer:
        metadata = {}

    layer = Layer()
    store = anchor_store(layer)
    assert anchor_store(layer) is store
    store.set(2, [[1, 2, 3], [4, 5, 6]])
    assert layer.metadata[METADATA_KEY] == {"2": [[1., 2., 3.], [4., 5., 6.]]}
    # metadata that arrives with a copy of the layer
    other = Layer()
    other.metadata = {METADATA_KEY: {"7": [[0., 0., 0.]]}}
    assert len(anchor_store(other).get(7)) == 1
    # replaced metadata
    layer.metadata = {}
    assert len(anchor_store(layer).get(2)) == 0
