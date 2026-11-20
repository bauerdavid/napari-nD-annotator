"""The experimental intensity prior of the minimal contour.

While it is on, the contour follows pixels whose intensity is similar to the intensity at the anchor points:
the speed of the path is the intensity similarity times the selected feature's speed to a small power.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.filters import sobel_h, sobel_v

from napari_nd_annotator.minimal_contour._eikonal_wrapper import MinimalContourCalculator
from napari_nd_annotator.minimal_contour.intensity_prior import IntensityPrior, feature_speed

H = W = 160
CENTER = np.array([80., 80.])
RADIUS = 50


def _on_circle(degrees):
    a = np.deg2rad(degrees)
    return CENTER + RADIUS * np.array([np.sin(a), np.cos(a)])


def _boundary_error(path):
    return np.abs(np.hypot(*(path - CENTER).T) - RADIUS).mean()


@pytest.fixture(scope="module")
def cell_image():
    """A cell (0.5) on background (0) with a bright nucleus (1.5) whose edge runs close to the cell's right side."""
    yy, xx = np.mgrid[:H, :W]
    img = np.zeros((H, W))
    img[np.hypot(yy - CENTER[0], xx - CENTER[1]) < RADIUS] = 0.5
    img[np.hypot(yy - CENTER[0], xx - CENTER[1] - 18) < 28] = 1.5
    img = gaussian_filter(img, 1.5) + np.random.default_rng(0).normal(0, 0.03, img.shape)
    return (img - img.min()) / (img.max() - img.min())


@pytest.fixture(scope="module")
def blob_image():
    """A blob with a soft falloff and a sharper bright core: the steepest edge is the core's, not any level of the blob.

    Returns the image and the line of 35% of its maximum (rows, columns) after the prior's smoothing.
    """
    yy, xx = np.mgrid[:H, :W]
    r2 = ((yy - 80) / 45.) ** 2 + ((xx - 80) / 60.) ** 2
    img = np.exp(-r2 * 1.5) + 0.8 * (np.hypot(yy - 80, xx - 95) < 20)
    img = gaussian_filter(img, 1.5) + np.random.default_rng(4).normal(0, 0.02, img.shape)
    img = (img - img.min()) / (img.max() - img.min())
    smooth = gaussian_filter(img, 1.)
    level_line = max(measure.find_contours(smooth, 0.35 * smooth.max()), key=len)
    return img, level_line


def _symmetric_gradient_calculator(img):
    calc = MinimalContourCalculator(None, 2)
    grad = np.hypot(sobel_v(img), sobel_h(img))
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    calc.set_method(2)
    calc.set_image(np.repeat(grad[..., None], 3, axis=2), np.empty((0, 0)), np.empty((0, 0)))
    calc.set_boundaries(0, 0, W, H)
    return calc


def _randers_calculator(img):
    calc = MinimalContourCalculator(None, 2)
    calc.set_image(np.repeat(img[..., None], 3, axis=2), sobel_v(img), sobel_h(img))
    calc.set_method(0)
    calc.set_boundaries(0, 0, W, H)
    return calc


def _path(calc, anchors):
    return calc.run(np.asarray(anchors, dtype=float), True, False, True)[0]


def _distance_to_line(path, line):
    """Mean distance of the path's points from the nearest point of a densely sampled line."""
    d = np.hypot(*(path[:, None, :] - line[None, :, :]).transpose(2, 0, 1))
    return d.min(axis=1).mean()


# IntensityPrior

def test_prior_needs_an_image_and_an_anchor(cell_image):
    prior = IntensityPrior()
    assert prior.reference([_on_circle(0)]) is None
    prior.set_image(cell_image)
    assert prior.reference(np.empty((0, 2))) is None
    assert prior.intensity_speed(np.empty((0, 2))) is None
    assert prior.speed_map(np.empty((0, 2))) is None


def test_reference_is_read_where_clicked():
    """The clicked pixel is sampled as it is, also right next to a strong edge."""
    img = np.zeros((30, 30))
    img[:, 15:] = 1.
    prior = IntensityPrior(smoothing=1.)
    prior.set_image(img)
    clicked = np.array([[10, 17], [20, 12]])
    assert np.allclose(prior.samples(clicked), prior.image[clicked[:, 0], clicked[:, 1]])


def test_snap_radius_moves_the_sample_to_the_edge():
    """With a snap radius a click a pixel inside a sharp edge samples the edge instead (kept for later use)."""
    img = np.zeros((30, 30))
    img[:, 15:] = 1.
    prior = IntensityPrior(smoothing=1., snap_radius=2)
    prior.set_image(img)
    clicked_inside = np.array([[10, 16], [20, 16]])
    # the edge is between columns 14 and 15: either pixel is on it
    edge_values = prior.image[10, 14:16]
    for sample in prior.samples(clicked_inside):
        assert np.isclose(edge_values, sample).any()
    assert prior.samples(clicked_inside).max() < prior.image[10, 16] - 0.1


def test_prior_works_from_a_single_anchor(blob_image):
    img, level_line = blob_image
    prior = IntensityPrior()
    assert prior.min_anchors == 1
    prior.set_image(img)
    click = level_line[0]
    mu, sigma = prior.reference([click])
    # the one sample is the reference; the tolerance is the lower limit: the noise floor, or the change over a pixel
    gradient = prior.grad_magnitude[prior._sample_indices([click])][0]
    assert sigma == max(prior.sigma_floor, gradient)
    assert np.isclose(mu, prior.samples([click])[0])
    # the whole level line is fast (its pixels are up to half a pixel off the level), the core and the background slow
    speed = prior.intensity_speed([click])
    on_line = np.round(level_line).astype(int)
    assert np.median(speed[on_line[:, 0], on_line[:, 1]]) > 0.6
    assert speed[80, 95] < 0.02 and speed[5, 5] < 0.02


@pytest.mark.parametrize("min_anchors, active_from", [(1, 1), (2, 2), (3, 3)])
def test_min_anchors(cell_image, min_anchors, active_from):
    prior = IntensityPrior(min_anchors=min_anchors)
    prior.set_image(cell_image)
    anchors = np.array([_on_circle(d) for d in (0, 30, 60)])
    for n in range(1, 4):
        assert (prior.reference(anchors[:n]) is not None) == (n >= active_from)


def test_min_anchors_is_at_least_one():
    prior = IntensityPrior(min_anchors=0)
    assert prior.min_anchors == 1
    prior.min_anchors = 2.0
    assert prior.min_anchors == 2


@pytest.mark.parametrize("max_slowdown", [3., 100., 1000.])
def test_intensity_speed_range(blob_image, max_slowdown):
    img, level_line = blob_image
    prior = IntensityPrior(max_slowdown=max_slowdown)
    prior.set_image(img)
    speed = prior.intensity_speed(level_line[[0, 40]], level_line[:41])
    assert speed.shape == img.shape
    assert speed.min() >= 1. / max_slowdown - 1e-9 and speed.max() <= 1.
    # the core and the background get the full slowdown
    assert speed[80, 95] < 1.01 / max_slowdown and speed[5, 5] < 1.01 / max_slowdown


def test_max_slowdown_is_clamped_and_one_switches_off(cell_image):
    prior = IntensityPrior(max_slowdown=1e6)
    assert prior.max_slowdown == 1000.
    prior.max_slowdown = 0.5
    assert prior.max_slowdown == 1.
    prior.set_image(cell_image)
    assert prior.intensity_speed([_on_circle(0)]) is None
    assert prior.speed_map([_on_circle(0)]) is None


def test_tolerance_floor_follows_the_noise():
    """The lower limit of sigma is a few times the image noise, unless it is set explicitly."""
    rng = np.random.default_rng(2)
    flat = np.full((100, 100), 0.5)
    floors = []
    for noise in (0.005, 0.02, 0.08):
        prior = IntensityPrior()
        prior.set_image(flat + rng.normal(0, noise, flat.shape))
        floors.append(prior.sigma_floor)
    # the floor grows with the noise, about in proportion (the prior smooths the image first)
    assert floors[0] < floors[1] < floors[2]
    assert 2.5 < floors[2] / floors[1] < 5.5
    prior.min_sigma = 0.05
    assert prior.sigma_floor == 0.05
    noise_free = IntensityPrior()
    noise_free.set_image(flat)
    assert noise_free.sigma_floor > 0


def test_match_type_is_validated():
    with pytest.raises(ValueError):
        IntensityPrior(match="ridge")


@pytest.fixture(scope="module")
def membrane_image():
    """A bright line (membrane) whose brightness varies along it, on a dark background."""
    yy, xx = np.mgrid[:80, :120]
    brightness = 0.5 + 0.4 * np.sin(xx / 15.)  # 0.1 .. 0.9 along the line
    img = np.where(np.abs(yy - 40) <= 1, brightness, 0.)
    return gaussian_filter(img, 1.) + np.random.default_rng(3).normal(0, 0.005, img.shape)


def test_similar_or_brighter_does_not_slow_down_the_brighter_parts(membrane_image):
    anchors = np.array([[40, 5], [40, 30]])
    contour = np.stack([np.full(26, 40), np.arange(5, 31)], axis=1)
    on_line = membrane_image[40] > np.median(membrane_image[40, 5:31])  # brighter than the traced part
    similar = IntensityPrior(match="similar")
    similar.set_image(membrane_image)
    brighter = IntensityPrior(match="similar or brighter")
    brighter.set_image(membrane_image)
    similar_speed = similar.intensity_speed(anchors, contour)
    brighter_speed = brighter.intensity_speed(anchors, contour)
    # two-sided: the brighter parts of the line are slowed down; one-sided: they keep their speed
    assert similar_speed[40][on_line].min() < 0.5
    assert brighter_speed[40][on_line].min() > 0.99
    # the dark background is slowed down either way
    assert brighter_speed[10, 60] < 0.02 and similar_speed[10, 60] < 0.02


def test_similar_or_darker_is_the_mirror_of_similar_or_brighter(membrane_image):
    anchors = np.array([[40, 5], [40, 30]])
    contour = np.stack([np.full(26, 40), np.arange(5, 31)], axis=1)
    brighter = IntensityPrior(match="similar or brighter")
    brighter.set_image(membrane_image)
    darker = IntensityPrior(match="similar or darker")
    darker.set_image(1. - membrane_image)
    assert np.allclose(brighter.intensity_speed(anchors, contour), darker.intensity_speed(anchors, contour))


def test_snapping_for_similar_or_brighter_takes_the_brightest_pixel(membrane_image):
    prior = IntensityPrior(match="similar or brighter", snap_radius=2)
    prior.set_image(membrane_image)
    sample = prior.samples(np.array([[42, 60]]))[0]
    assert np.isclose(sample, prior.image[40:45, 58:63].max())


def test_inconsistent_boundary_widens_the_tolerance():
    """When the traced boundary intensities disagree (e.g. DIC), sigma grows and the prior has little effect."""
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 1, (50, 50))
    prior = IntensityPrior(smoothing=0)
    prior.set_image(img)
    contour = np.stack([np.full(40, 25), np.arange(5, 45)], axis=1)
    _, sigma = prior.reference(contour[[0, -1]], contour)
    assert sigma > 0.2


def test_prior_averages_channels():
    prior = IntensityPrior(smoothing=0)
    prior.set_image(np.stack([np.zeros((5, 5)), np.ones((5, 5)), np.full((5, 5), 0.5)], axis=-1))
    assert np.allclose(prior.image, 0.5)


def test_feature_speed():
    rng = np.random.default_rng(5)
    features = rng.uniform(0, 1, (40, 50, 3))
    # without gradients: the first channel, its 1st..99th percentile scaled to 0..1
    speed = feature_speed(features)
    low, high = np.percentile(features[..., 0], (1, 99))
    assert np.allclose(speed, np.clip((features[..., 0] - low) / (high - low), 0, 1))
    # with gradients: the gradient magnitude
    grad_x, grad_y = rng.normal(size=(2, 40, 50))
    magnitude = np.hypot(grad_x, grad_y)
    low, high = np.percentile(magnitude, (1, 99))
    assert np.allclose(feature_speed(features, grad_x, grad_y), np.clip((magnitude - low) / (high - low), 0, 1))
    assert np.array_equal(feature_speed(np.full((5, 5), 0.3)), np.ones((5, 5)))


def test_speed_map_mixes_in_the_feature(blob_image):
    img, level_line = blob_image
    feature = np.random.default_rng(6).uniform(0, 1, img.shape)
    anchors = level_line[[0]]
    prior = IntensityPrior(feature_weight=0.25)
    prior.set_image(img)
    intensity = prior.intensity_speed(anchors)
    assert np.allclose(prior.speed_map(anchors, None, feature), intensity * np.maximum(feature, 1e-3) ** 0.25)
    # no feature, or no weight: the intensity alone
    assert np.array_equal(prior.speed_map(anchors), intensity)
    prior.feature_weight = 0.
    assert np.array_equal(prior.speed_map(anchors, None, feature), intensity)
    prior.feature_weight = 5.
    assert prior.feature_weight == 1.


# MinimalContourCalculator.set_speed_map

@pytest.mark.parametrize("make_calculator", [_randers_calculator, _symmetric_gradient_calculator])
def test_without_a_speed_map_the_metric_is_unchanged(cell_image, make_calculator):
    calc = make_calculator(cell_image)
    anchors = [_on_circle(70), _on_circle(-70)]
    base = _path(calc, anchors)
    calc.set_speed_map(np.random.default_rng(7).uniform(0.1, 1, cell_image.shape))
    assert not np.array_equal(_path(calc, anchors), base)
    calc.set_speed_map(None)
    assert np.array_equal(_path(calc, anchors), base)


@pytest.mark.parametrize("make_calculator", [_randers_calculator, _symmetric_gradient_calculator])
def test_uniform_speed_map_gives_a_shortest_path(cell_image, make_calculator):
    """The speed map replaces the metric: with the same speed everywhere the image (and its edges) do not matter.

    On the pixel grid the path is one of the equally short 8-connected paths, not necessarily the straight line.
    """
    calc = make_calculator(cell_image)
    a, b = np.array([20., 30.]), np.array([130., 120.])
    calc.set_speed_map(np.full(cell_image.shape, 0.5))
    path = _path(calc, [a, b])
    length = np.hypot(*np.diff(path, axis=0).T).sum()
    assert length < 1.1 * np.hypot(*(b - a))


def test_speed_map_scale_does_not_matter(cell_image):
    calc = _symmetric_gradient_calculator(cell_image)
    anchors = [_on_circle(70), _on_circle(-70)]
    speed = np.random.default_rng(8).uniform(0.1, 1, cell_image.shape)
    calc.set_speed_map(speed)
    path = _path(calc, anchors)
    calc.set_speed_map(speed * 7.)
    assert np.array_equal(_path(calc, anchors), path)


def test_speed_map_is_validated(cell_image):
    calc = _symmetric_gradient_calculator(cell_image)
    with pytest.raises(ValueError):
        calc.set_speed_map(np.ones((10, 10)))
    with pytest.raises(ValueError):
        calc.set_speed_map(np.zeros(cell_image.shape))


def test_set_image_drops_the_speed_map(cell_image):
    calc = _randers_calculator(cell_image)
    anchors = [_on_circle(70), _on_circle(-70)]
    base = _path(calc, anchors)
    # only the left half of the image is fast: the path has to go around the left side
    calc.set_speed_map(np.where(np.arange(W) < 80, 1., 0.05)[None].repeat(H, 0))
    assert not np.array_equal(_path(calc, anchors), base)
    calc.set_image(np.repeat(cell_image[..., None], 3, axis=2), sobel_v(cell_image), sobel_h(cell_image))
    calc.set_boundaries(0, 0, W, H)
    assert np.array_equal(_path(calc, anchors), base)


@pytest.mark.parametrize("max_slowdown", [10., 100., 1000.])
@pytest.mark.parametrize("feature_weight", [0., 0.25])
@pytest.mark.parametrize("make_calculator", [_randers_calculator, _symmetric_gradient_calculator])
def test_contour_follows_the_clicked_intensity_level(blob_image, make_calculator, feature_weight, max_slowdown):
    """Clicked on the 35% level of a soft blob, the contour follows that level and not the blob's steepest edge."""
    img, level_line = blob_image
    calc = make_calculator(img)
    n = len(level_line)
    a, b = level_line[0], level_line[n // 4]
    segment_line = level_line[:n // 4 + 1]
    without = _distance_to_line(_path(calc, [b, a]), segment_line)
    prior = IntensityPrior(max_slowdown=max_slowdown, feature_weight=feature_weight)
    prior.set_image(img)
    if make_calculator is _randers_calculator:
        feature = feature_speed(img, sobel_v(img), sobel_h(img))
    else:
        feature = feature_speed(np.hypot(sobel_v(img), sobel_h(img)))
    calc.set_speed_map(prior.speed_map([a], None, feature))
    path = _path(calc, [b, a])
    # the path tracing reached the start point instead of running into its step limit
    assert len(path) < 1000
    with_prior = _distance_to_line(path, segment_line)
    assert with_prior < 1.
    assert with_prior < without


# widget and overlay

def _process_events():
    from qtpy.QtWidgets import QApplication
    QApplication.processEvents()


@pytest.fixture
def viewer():
    import napari
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()
    _process_events()


def _scene(viewer, monkeypatch, env=None):
    from napari_nd_annotator import AnnotatorWidget
    from napari_nd_annotator._helper_functions import viewer_qt_viewer
    from napari_nd_annotator._widgets.minimal_contour_widget import INTENSITY_PRIOR_ENV
    if env is None:
        monkeypatch.delenv(INTENSITY_PRIOR_ENV, raising=False)
    else:
        monkeypatch.setenv(INTENSITY_PRIOR_ENV, env)
    annotator = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(annotator, name="Annotator")
    data = (np.random.default_rng(0).uniform(0, 1, (60, 70)) * 1000).astype(np.uint16)
    image = viewer.add_image(data, name="image")
    labels = viewer.add_labels(np.zeros(data.shape, dtype=np.uint16), name="labels", metadata={"source_image_layer": image})
    viewer.layers.selection.active = labels
    _process_events()
    return SimpleNamespace(mc=annotator.ToolsWidget.minimal_contour_widget, image=image, labels=labels,
                           qt_viewer=viewer_qt_viewer(viewer))


def _vispy_overlay(scene):
    scene.qt_viewer.controls.widgets[scene.labels].mc_button.click()
    _process_events()
    overlay = scene.labels._overlays["minimal_contour"]
    visual_map = getattr(scene.qt_viewer.canvas, "_layer_overlay_to_visual", None)
    if visual_map is None:
        pytest.skip("this napari version keeps overlay visuals elsewhere")
    return overlay, visual_map[scene.labels][overlay]


def _spy_speed_maps(monkeypatch):
    """The arguments of every set_speed_map call of the overlays' calculator."""
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
    calls = []
    calculator = VispyMinimalContourOverlay.mc_calculator

    class CalculatorSpy:
        def set_speed_map(self, speed):
            calls.append(speed)
            calculator.set_speed_map(speed)

        def __getattr__(self, name):
            return getattr(calculator, name)

    monkeypatch.setattr(VispyMinimalContourOverlay, "mc_calculator", CalculatorSpy())
    return calls


@pytest.fixture(autouse=False)
def restore_prior_state():
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
    yield
    VispyMinimalContourOverlay.set_use_intensity_prior(False)
    VispyMinimalContourOverlay.set_intensity_prior_min_anchors(1)
    VispyMinimalContourOverlay.set_intensity_prior_max_slowdown(100.)
    VispyMinimalContourOverlay.set_intensity_prior_feature_weight(0.25)
    VispyMinimalContourOverlay.set_intensity_prior_match("similar")


def _shown(widget):
    # .visible reports whether it is on screen, which it never is with a hidden viewer
    return not widget.native.isHidden()


def test_controls_are_hidden_by_default(viewer, monkeypatch, restore_prior_state):
    from napari_nd_annotator._widgets.minimal_contour_widget import INTENSITY_PRIOR_FIELDS
    mc = _scene(viewer, monkeypatch).mc
    for name in INTENSITY_PRIOR_FIELDS:
        assert not _shown(getattr(mc, name))
    assert _shown(mc.used_feature_combobox)


@pytest.mark.parametrize("value", ["", "0"])
def test_controls_stay_hidden_for_empty_or_zero(viewer, monkeypatch, restore_prior_state, value):
    mc = _scene(viewer, monkeypatch, env=value).mc
    assert not _shown(mc.intensity_prior_checkbox)


def test_environment_variable_shows_controls_that_are_not_persisted(viewer, monkeypatch, restore_prior_state):
    from napari_nd_annotator._widgets.minimal_contour_widget import INTENSITY_PRIOR_FIELDS
    mc = _scene(viewer, monkeypatch, env="1").mc
    for name in INTENSITY_PRIOR_FIELDS:
        assert _shown(getattr(mc, name))
    mc.intensity_prior_checkbox.value = True
    mc.intensity_prior_min_anchors_spinbox.value = 3
    mc.intensity_prior_slowdown_combobox.value = 1000.
    mc.intensity_prior_match_combobox.value = "similar or brighter"
    mc.intensity_prior_feature_weight_slider.value = 0.6
    state = mc.__magicclass_serialize__()
    for key in INTENSITY_PRIOR_FIELDS:
        assert key not in state
        assert key not in state["Image Features"]


def test_controls_set_the_prior(viewer, monkeypatch, restore_prior_state):
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
    mc = _scene(viewer, monkeypatch, env="1").mc
    prior = VispyMinimalContourOverlay.intensity_prior
    # the defaults of the controls are the prior's
    assert prior.max_slowdown == mc.intensity_prior_slowdown_combobox.value == 100.
    assert prior.match == mc.intensity_prior_match_combobox.value == "similar"
    assert prior.feature_weight == mc.intensity_prior_feature_weight_slider.value == 0.25
    assert prior.snap_radius == 0
    mc.intensity_prior_slowdown_combobox.value = 1000.
    mc.intensity_prior_match_combobox.value = "similar or brighter"
    mc.intensity_prior_feature_weight_slider.value = 0.5
    assert prior.max_slowdown == 1000. and prior.match == "similar or brighter" and prior.feature_weight == 0.5
    assert mc.intensity_prior_slowdown_combobox.native.currentText() == "×1000"


def test_low_intensity_feature_gives_the_prior_the_displayed_intensities(viewer, monkeypatch, restore_prior_state):
    """The low intensity feature inverts the image; the prior still compares the intensities as displayed."""
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
    scene = _scene(viewer, monkeypatch, env="1")
    scene.mc.used_feature_combobox.value = "Low intensity"
    _vispy_overlay(scene)
    raw = np.asarray(VispyMinimalContourOverlay.intensity_prior._raw_image, dtype=float)
    raw = raw.mean(-1) if raw.ndim == 3 else raw
    assert np.corrcoef(raw.ravel(), scene.image.data.astype(float).ravel())[0, 1] > 0.99


def test_overlay_sets_the_speed_map_from_accepted_anchors(viewer, monkeypatch, restore_prior_state):
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
    scene = _scene(viewer, monkeypatch, env="1")
    overlay, vispy_overlay = _vispy_overlay(scene)
    prior = VispyMinimalContourOverlay.intensity_prior
    assert prior.shape == scene.image.data.shape
    # nothing is computed for the prior while it is off
    assert prior._image is None and VispyMinimalContourOverlay._feature_speed is None

    calls = _spy_speed_maps(monkeypatch)
    overlay.anchor_points = [[10., 10.], [10., 40.], [30., 30.]]  # the last one follows the cursor
    overlay.stored_contour = [[10., x] for x in range(10, 41)]

    scene.mc.intensity_prior_checkbox.value = True
    assert calls[-1] is not None and calls[-1].shape == scene.image.data.shape
    # the speed map mixes in the selected feature
    feature = VispyMinimalContourOverlay._feature_speed
    assert feature is not None
    anchors, contour = np.array([[10., 10.], [10., 40.]]), np.array(overlay.stored_contour)
    assert np.allclose(calls[-1], prior.speed_map(anchors, contour, feature))
    scene.mc.intensity_prior_feature_weight_slider.value = 0.
    assert np.allclose(calls[-1], prior.intensity_speed(anchors, contour))

    overlay.anchor_points = [[10., 10.], [30., 30.]]  # one accepted anchor: enough by default
    overlay.stored_contour = []
    vispy_overlay.update_speed_map()
    assert calls[-1] is not None

    scene.mc.intensity_prior_min_anchors_spinbox.value = 2  # recomputes: one anchor is not enough any more
    assert calls[-1] is None
    overlay.anchor_points = [[10., 10.], [10., 40.], [30., 30.]]
    vispy_overlay.update_speed_map()
    assert calls[-1] is not None
    scene.mc.intensity_prior_min_anchors_spinbox.value = 3
    assert calls[-1] is None
    scene.mc.intensity_prior_min_anchors_spinbox.value = 1
    assert calls[-1] is not None

    overlay.anchor_points = []  # no anchor at all
    vispy_overlay.update_speed_map()
    assert calls[-1] is None
    overlay.anchor_points = [[10., 10.], [10., 40.], [30., 30.]]
    vispy_overlay.update_speed_map()
    vispy_overlay.clear(None)
    assert calls[-1] is None

    overlay.anchor_points = [[10., 10.], [10., 40.], [30., 30.]]
    scene.mc.intensity_prior_checkbox.value = False
    assert calls[-1] is None


def test_overlay_samples_the_roi_window_in_window_coordinates(viewer, monkeypatch, restore_prior_state):
    """With a ROI the prior's image is the window; anchors (plane coordinates) are shifted into it."""
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
    scene = _scene(viewer, monkeypatch, env="1")
    overlay, vispy_overlay = _vispy_overlay(scene)

    # a plane of the scene's shape: dark left of column 45, bright from it; the calculator has columns 30..70
    plane_shape = scene.image.data.shape
    plane = np.where(np.arange(plane_shape[1]) < 45, 0.2, 0.8)[None].repeat(plane_shape[0], 0)
    window = (slice(0, plane_shape[0]), slice(30, plane_shape[1]))
    VispyMinimalContourOverlay.set_prior_image(plane[window])
    VispyMinimalContourOverlay.set_calculator_feature(np.repeat(plane[window][..., None], 3, axis=2),
                                                      roi_origin=(0, 30), plane_shape=plane_shape)

    calls = _spy_speed_maps(monkeypatch)
    scene.mc.intensity_prior_feature_weight_slider.value = 0.  # the intensity alone
    scene.mc.intensity_prior_checkbox.value = True
    # an anchor on the dark part, at plane column 31 (window column 1), and the cursor; one outside the window
    overlay.anchor_points = [[30., 10.], [30., 31.], [40., 60.]]
    vispy_overlay.update_speed_map()
    speed = calls[-1]
    assert speed is not None and speed.shape == plane[window].shape
    # the dark columns are fast: the anchor was sampled at the right place (unshifted it would be bright)
    assert speed[30, 1] > 0.95
    assert speed[30, 30] < 0.3
    # the anchor outside the window is not sampled: with it alone there is no reference
    overlay.anchor_points = [[30., 10.], [40., 60.]]
    vispy_overlay.update_speed_map()
    assert calls[-1] is None
    VispyMinimalContourOverlay.set_calculator_feature(None)
