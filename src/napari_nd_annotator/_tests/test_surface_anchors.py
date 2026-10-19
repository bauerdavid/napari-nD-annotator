"""Tests and accuracy evaluation of the surface anchor suggestion (pure numpy, no viewer).

The distance reported by ``evaluate`` is relative to the true long diameter; rotated ellipsoids lose some of it
because the rays follow the (axis aligned) box axes.

Run ``python -m napari_nd_annotator._tests.test_surface_anchors`` to print the surface error statistics of the
evaluation over random ellipsoids (and of the scikit-image ``cells3d`` nuclei if the data is available offline).
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

from napari_nd_annotator.surface_anchors import suggest_surface_anchors

ALPHA = .01
BETA = 5.


def edge_cost(image, alpha=ALPHA, beta=BETA, sigma=1.):
    """Edge cost like the minimal surface widget with the gradient feature: low on edges, high in flat regions."""
    image = (image - image.min()) / (image.max() - image.min())
    features = gaussian_gradient_magnitude(image, sigma)
    alpha = (alpha - np.exp(-beta)) / (1 - np.exp(-beta))
    phi = alpha + (1 - alpha) * np.exp(-beta * features)
    return phi / phi.max()


class Ellipsoid:
    """Ellipsoid in physical coordinates; ``centre`` and ``semi_axes`` are physical, ``rotation`` a 3x3 matrix."""

    def __init__(self, centre, semi_axes, rotation=None):
        self.centre = np.asarray(centre, dtype=float)
        self.semi_axes = np.asarray(semi_axes, dtype=float)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)

    def level(self, physical_points):
        """Implicit function: 1 on the surface, <1 inside. Returns it and its physical gradient norm."""
        u = (np.asarray(physical_points) - self.centre) @ self.rotation
        scaled = u / self.semi_axes
        f = np.sqrt((scaled ** 2).sum(-1))
        grad = np.sqrt(((u / self.semi_axes ** 2) ** 2).sum(-1)) / np.maximum(f, 1e-12)
        return f, grad

    def distance_to_surface(self, physical_points):
        """First order approximation of the euclidean distance to the surface (physical units)."""
        f, grad = self.level(physical_points)
        return np.abs(f - 1) / grad

    def mask(self, shape, scale):
        grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, scale)], indexing="ij"), -1)
        return self.level(grid)[0] <= 1


def render(ellipsoids, shape, scale, noise=0., intensities=None, seed=0):
    """Image with bright ellipsoids on a dark background, blurred like a microscope and with gaussian noise."""
    scale = np.asarray(scale, dtype=float)
    image = np.zeros(shape)
    for i, ellipsoid in enumerate(ellipsoids):
        image[ellipsoid.mask(shape, scale)] = 1. if intensities is None else intensities[i]
    image = gaussian_filter(image, scale.min() / scale)
    return image + np.random.default_rng(seed).normal(0, noise, shape)


def bbox_of(mask):
    coords = np.argwhere(mask)
    return coords.min(0).astype(float), coords.max(0).astype(float)


def surface_error(ellipsoid, points, scale):
    """Distance of the points from the ellipsoid surface in voxels of the smallest voxel size."""
    scale = np.asarray(scale, dtype=float)
    return ellipsoid.distance_to_surface(np.asarray(points) * scale) / scale.min()


def check_suggestion(result, ellipsoid, bbox, scale, long_axis, tolerance=2.5):
    assert result.found, result.reason
    scale = np.asarray(scale, dtype=float)
    assert result.axis == long_axis
    for p in (result.p1, result.p2):
        assert np.all(p >= bbox[0]) and np.all(p <= bbox[1])
    errors = surface_error(ellipsoid, [result.p1, result.p2], scale)
    assert np.all(errors < tolerance), errors
    # antipodal: roughly the full length of the object, along the long axis
    full_length = 2 * ellipsoid.semi_axes.max()
    assert result.distance > .85 * full_length
    direction = (result.p2 - result.p1) * scale
    assert abs(direction[long_axis]) / np.linalg.norm(direction) > .95
    assert result.p1[long_axis] < result.p2[long_axis]


@pytest.mark.parametrize("scale", [(1., 1., 1.), (2., 1., 1.), (1., 1., 2.)])
@pytest.mark.parametrize("long_axis", [0, 1, 2])
def test_ellipsoid_long_axis(scale, long_axis):
    shape = (60, 70, 80)
    scale = np.asarray(scale)
    semi = np.array([10., 10., 10.]) * scale.min()
    semi[long_axis] = 22 * scale.min()
    # off-centre in the image
    centre = np.array([27, 33, 36]) * scale
    ellipsoid = Ellipsoid(centre, semi)
    image = render([ellipsoid], shape, scale, noise=.05)
    bbox = bbox_of(ellipsoid.mask(shape, scale))
    # make sure the long axis in physical space is also the longest box extent
    assert np.argmax((bbox[1] - bbox[0]) * scale) == long_axis
    result = suggest_surface_anchors(*bbox, edge_cost(image), scale)
    check_suggestion(result, ellipsoid, bbox, scale, long_axis)
    # diagnostics: boundary is far below the flat level
    assert max(result.phi_ratio) < .8
    assert result.confidence > .3
    assert min(result.zscore) > 4


def test_off_centre_and_loose_bbox():
    # object not in the middle of a generous box: the box centre is off the object centre
    shape = (60, 60, 60)
    ellipsoid = Ellipsoid((28, 30, 26), (9, 9, 18))
    image = render([ellipsoid], shape, (1, 1, 1), noise=.05)
    bbox = (np.array([16., 16., 5.]), np.array([40., 44., 50.]))
    result = suggest_surface_anchors(*bbox, edge_cost(image))
    assert result.found, result.reason
    errors = surface_error(ellipsoid, [result.p1, result.p2], (1, 1, 1))
    assert np.all(errors < 2.5), errors
    assert result.axis == 2


def test_touching_neighbour():
    # a second object touches the first one on the side; its boundary must not attract the anchors
    shape = (60, 70, 60)
    scale = np.array([1., 1., 1.])
    obj = Ellipsoid((30, 25, 30), (8, 18, 8))
    neighbour = Ellipsoid((30, 25 + 18 + 9 + 1, 30), (9, 9, 9))
    image = render([obj, neighbour], shape, scale, noise=.05)
    bbox = bbox_of(obj.mask(shape, scale))
    # a slightly loose box so the neighbour is partly inside
    bbox = (bbox[0] - 2, bbox[1] + 4)
    result = suggest_surface_anchors(*bbox, edge_cost(image), scale)
    check_suggestion(result, obj, (bbox[0], bbox[1]), scale, 1)


def test_flat_phi_gives_no_suggestion():
    phi = np.ones((30, 30, 30))
    result = suggest_surface_anchors([5, 5, 5], [25, 25, 25], phi)
    assert not result.found
    assert not result
    assert result.p1 is None and result.p2 is None
    assert result.reason


def test_numerically_flat_phi_gives_no_suggestion():
    rng = np.random.default_rng(1)
    for _ in range(10):
        phi = 1 - np.abs(rng.normal(0, 1e-5, (40, 40, 40)))
        result = suggest_surface_anchors([5, 5, 5], [35, 35, 35], phi)
        assert not result.found


def test_object_free_image_has_weak_confidence():
    # Known limitation: the edge cost of a pure noise image is not flat, so a suggestion is made, but its
    # confidence is far below that of a real object (about 0.45 in the other tests)
    rng = np.random.default_rng(1)
    for _ in range(5):
        phi = edge_cost(rng.normal(0, 1, (40, 40, 40)))
        result = suggest_surface_anchors([5, 5, 5], [35, 35, 35], phi)
        assert result.confidence < .15


def test_box_outside_or_degenerate():
    phi = np.ones((10, 10, 10))
    assert not suggest_surface_anchors([20, 20, 20], [30, 30, 30], phi).found
    assert not suggest_surface_anchors([4, 4, 4], [4, 4, 4], phi).found
    with pytest.raises(ValueError):
        suggest_surface_anchors([0, 0], [5, 5], phi)


def test_tight_box_wall_on_surface():
    # the surface touches the box wall; the points must still be inside the box
    shape = (40, 40, 60)
    ellipsoid = Ellipsoid((20, 20, 30), (8, 8, 22))
    image = render([ellipsoid], shape, (1, 1, 1), noise=.02)
    bbox = bbox_of(ellipsoid.mask(shape, (1, 1, 1)))
    result = suggest_surface_anchors(*bbox, edge_cost(image))
    check_suggestion(result, ellipsoid, bbox, (1, 1, 1), 2, tolerance=2.)


def test_axis_strategies():
    shape = (50, 50, 50)
    ellipsoid = Ellipsoid((25, 25, 25), (8, 12, 20))
    phi = edge_cost(render([ellipsoid], shape, (1, 1, 1), noise=.03))
    bbox = bbox_of(ellipsoid.mask(shape, (1, 1, 1)))
    for strategy in ("longest", "fallback", "best"):
        result = suggest_surface_anchors(*bbox, phi, axis_strategy=strategy)
        assert result.found and result.axis == 2
    with pytest.raises(ValueError):
        suggest_surface_anchors(*bbox, phi, axis_strategy="nope")


def test_fallback_to_other_axis():
    # the box cuts the object: no boundary along its longest axis, but the lateral surface is inside the box
    shape = (60, 60, 80)
    ellipsoid = Ellipsoid((30, 30, 40), (10, 10, 30))
    phi = edge_cost(render([ellipsoid], shape, (1, 1, 1), noise=.03))
    bbox = (np.array([17., 17., 22.]), np.array([43., 43., 58.]))
    assert not suggest_surface_anchors(*bbox, phi, axis_strategy="longest").found
    result = suggest_surface_anchors(*bbox, phi)
    assert result.found and result.axis in (0, 1)
    assert np.all(surface_error(ellipsoid, [result.p1, result.p2], (1, 1, 1)) < 2.5)


def random_ellipsoid_case(rng, noise):
    """Random rotated, anisotropically scaled ellipsoid in a random image; returns phi, bbox, ellipsoid, scale."""
    scale = np.array([rng.choice([1., 2., 3.]), 1., 1.])
    shape = (int(60 / scale[0]) + 20, 80, 80)
    semi = rng.uniform(7, 20, 3)
    if semi.max() / semi.min() < 1.3:
        semi[0] *= 1.4
    # random orthogonal matrix (QR of a gaussian matrix) for half of the objects
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0] if rng.random() < .5 else None
    centre = (np.array(shape) / 2 + rng.uniform(-5, 5, 3)) * scale
    ellipsoid = Ellipsoid(centre, semi, rotation)
    ellipsoids = [ellipsoid]
    intensities = [1.]
    if rng.random() < .5:
        # touching neighbour on a random side
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = rng.uniform(6, 12)
        # place along the direction until the surfaces are about one voxel apart
        offset = 0.
        while ellipsoid.level((centre + direction * offset)[None])[0][0] < 1:
            offset += .5
        ellipsoids.append(Ellipsoid(centre + direction * (offset + radius + 1.5), (radius,) * 3))
        intensities.append(rng.uniform(.6, 1.))
    mask = ellipsoid.mask(shape, scale)
    image = render(ellipsoids, shape, scale, noise, intensities, seed=int(rng.integers(1 << 30)))
    bbox = bbox_of(mask)
    # the user's box is rarely tight: random margin of up to 3 voxels (per side and axis)
    margin = rng.integers(0, 4, (2, 3))
    bbox = (np.maximum(bbox[0] - margin[0], 0), np.minimum(bbox[1] + margin[1], np.array(shape) - 1))
    return edge_cost(image), bbox, ellipsoid, scale


def evaluate(n_cases=50, noise=.1, seed=0, **kwargs):
    """Surface error (voxels of the smallest voxel size) of the suggested points for random ellipsoids."""
    rng = np.random.default_rng(seed)
    errors = []
    lengths = []
    missing = 0
    for _ in range(n_cases):
        phi, bbox, ellipsoid, scale = random_ellipsoid_case(rng, noise)
        result = suggest_surface_anchors(*bbox, phi, scale, **kwargs)
        if not result.found:
            missing += 1
            continue
        errors.extend(surface_error(ellipsoid, [result.p1, result.p2], scale))
        lengths.append(result.distance / (2 * ellipsoid.semi_axes.max()))
    return dict(
        n_cases=n_cases, missing=missing, errors=np.asarray(errors), length_ratio=np.asarray(lengths)
    )


def summarize(name, stats):
    errors = stats["errors"]
    text = (
        "%s: %d cases, no suggestion %d | surface error [voxels] median %.2f, mean %.2f, p90 %.2f, max %.2f, "
        "fraction < 2: %.2f, fraction < 3: %.2f" % (
            name, stats["n_cases"], stats["missing"], np.median(errors), errors.mean(), np.percentile(errors, 90),
            errors.max(), (errors < 2).mean(), (errors < 3).mean())
    )
    if "length_ratio" in stats:
        text += " | distance / true long diameter median %.2f (min %.2f)" % (
            np.median(stats["length_ratio"]), stats["length_ratio"].min())
    return text


@pytest.mark.parametrize("noise", [.05, .1])
def test_random_ellipsoid_evaluation(noise):
    stats = evaluate(n_cases=50, noise=noise)
    print(summarize("noise %.2f" % noise, stats))
    assert stats["missing"] <= 2
    assert np.median(stats["errors"]) < 1.5
    assert (stats["errors"] < 3).mean() > .9
    assert np.median(stats["length_ratio"]) > .9


def cells3d_evaluation(min_volume=500, **kwargs):
    """Anchor error on the nuclei of scikit-image ``cells3d``, or None if the data is not cached (never downloads).

    There is no ground truth, so the nuclei are segmented with Otsu + a distance transform watershed; the error of
    a point is its distance (voxels of the smallest voxel size) from the surface of its own segment.
    """
    import os
    from scipy import ndimage as ndi
    from skimage import data
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.measure import regionprops
    from skimage.segmentation import watershed
    if not os.path.exists(os.path.join(data.data_dir, "cells3d.tif")):
        return None
    nuclei = data.cells3d()[:, 1].astype(float)
    scale = np.array([.29, .26, .26])
    scale = scale / scale.min()
    smooth = ndi.gaussian_filter(nuclei, 1)
    foreground = smooth > threshold_otsu(smooth)
    distance = ndi.distance_transform_edt(foreground, sampling=scale)
    peaks = peak_local_max(ndi.gaussian_filter(distance, 1), min_distance=8, labels=ndi.label(foreground)[0])
    markers = np.zeros(nuclei.shape, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-distance, markers, mask=foreground)
    phi = edge_cost(nuclei)
    errors = []
    n_objects = 0
    n_found = 0
    for region in regionprops(labels):
        if region.area < min_volume:
            continue
        n_objects += 1
        bbox = np.array(region.bbox)
        # the user's box is a bit loose
        lower = np.maximum(bbox[:3] - 2, 0).astype(float)
        upper = np.minimum(bbox[3:] + 1, np.array(nuclei.shape) - 1).astype(float)
        result = suggest_surface_anchors(lower, upper, phi, scale, **kwargs)
        if not result.found:
            continue
        n_found += 1
        crop_lo = np.maximum(bbox[:3] - 25, 0)
        crop = tuple(slice(a, b) for a, b in zip(crop_lo, np.minimum(bbox[3:] + 25, nuclei.shape)))
        mask = labels[crop] == region.label
        signed_distance = (ndi.distance_transform_edt(mask, sampling=scale)
                           - ndi.distance_transform_edt(~mask, sampling=scale))
        for point in (result.p1, result.p2):
            errors.append(abs(ndi.map_coordinates(
                signed_distance, (point - crop_lo)[:, None], order=1, mode="nearest")[0]))
    return dict(n_cases=n_objects, missing=n_objects - n_found, errors=np.asarray(errors))


if __name__ == "__main__":
    for noise in (.05, .1, .2):
        print(summarize("ellipsoids, noise %.2f" % noise, evaluate(noise=noise)))
    nuclei_stats = cells3d_evaluation()
    if nuclei_stats is None:
        print("cells3d is not cached locally, skipped")
    else:
        print(summarize("cells3d nuclei", nuclei_stats))
