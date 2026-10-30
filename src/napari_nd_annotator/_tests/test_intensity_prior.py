"""The experimental intensity prior of the minimal contour.

It slows the minimal path down on pixels whose intensity differs from the already traced boundary.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel_h, sobel_v

from napari_nd_annotator.minimal_contour._eikonal_wrapper import MinimalContourCalculator
from napari_nd_annotator.minimal_contour.intensity_prior import IntensityPrior

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


def _traced_left_side():
    """Anchors and accepted contour of a boundary part away from the nucleus."""
    anchors = np.array([_on_circle(250), _on_circle(290)])
    contour = np.array([_on_circle(d) for d in np.linspace(250, 290, 60)])
    return anchors, contour


# IntensityPrior

def test_prior_needs_an_image_and_an_anchor(cell_image):
    prior = IntensityPrior()
    assert prior.reference([_on_circle(0)]) is None
    prior.set_image(cell_image)
    assert prior.reference(np.empty((0, 2))) is None
    assert prior.speed_modulation(np.empty((0, 2))) is None


def test_prior_works_from_a_single_anchor(cell_image):
    prior = IntensityPrior(strength=0.8)
    assert prior.min_anchors == 1
    prior.set_image(cell_image)
    mu, sigma = prior.reference([_on_circle(0)])
    # the one sample is the reference, the tolerance is the lower limit
    assert sigma == prior.min_sigma
    assert np.isclose(mu, prior.samples([_on_circle(0)])[0])
    modulation = prior.speed_modulation([_on_circle(0)])
    # away from the nucleus, every radius crosses the reference intensity within a few pixels of the cell boundary
    for degrees in (100, 140, 180, 220, 260):
        direction = np.array([np.sin(np.deg2rad(degrees)), np.cos(np.deg2rad(degrees))])
        across = np.round(CENTER + np.outer(RADIUS + np.arange(-3, 4), direction)).astype(int)
        assert modulation[across[:, 0], across[:, 1]].max() > 0.9
    assert modulation[80, 80] < 0.4 and modulation[5, 5] < 0.4


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


def test_prior_modulation_range(cell_image):
    prior = IntensityPrior(strength=0.7)
    prior.set_image(cell_image)
    modulation = prior.speed_modulation(*_traced_left_side())
    assert modulation.shape == cell_image.shape
    assert modulation.min() >= 0.3 - 1e-9 and modulation.max() <= 1.
    # the cell boundary keeps its speed, the nucleus edge and the background are slowed down
    boundary = np.round(_on_circle(0)).astype(int)
    assert modulation[tuple(boundary)] > 0.9
    assert modulation[80, 80] < 0.4 and modulation[5, 5] < 0.4


def test_prior_strength_is_clamped_and_zero_switches_off(cell_image):
    prior = IntensityPrior(strength=2.)
    assert prior.strength == 0.95
    prior.strength = 0
    prior.set_image(cell_image)
    assert prior.speed_modulation(*_traced_left_side()) is None


def test_anchor_is_sampled_on_the_edge():
    """A click a pixel inside a sharp edge samples the edge, not the inside of the object."""
    img = np.zeros((30, 30))
    img[:, 15:] = 1.
    prior = IntensityPrior(smoothing=1.)
    prior.set_image(img)
    clicked_inside = np.array([[10, 16], [20, 16]])
    # the edge is between columns 14 and 15: either pixel is on it
    edge_values = prior.image[10, 14:16]
    for sample in prior.samples(clicked_inside):
        assert np.isclose(edge_values, sample).any()
    assert prior.samples(clicked_inside).max() < prior.image[10, 16] - 0.1


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


# MinimalContourCalculator.set_speed_modulation

@pytest.mark.parametrize("make_calculator", [_randers_calculator, _symmetric_gradient_calculator])
def test_all_ones_modulation_changes_nothing(cell_image, make_calculator):
    calc = make_calculator(cell_image)
    anchors = [_on_circle(70), _on_circle(-70)]
    base = _path(calc, anchors)
    calc.set_speed_modulation(np.ones_like(cell_image))
    assert np.array_equal(_path(calc, anchors), base)
    calc.set_speed_modulation(None)
    assert np.array_equal(_path(calc, anchors), base)


def test_modulation_shape_must_match_the_image(cell_image):
    calc = _symmetric_gradient_calculator(cell_image)
    with pytest.raises(ValueError):
        calc.set_speed_modulation(np.ones((10, 10)))


def test_set_image_drops_the_modulation(cell_image):
    calc = _randers_calculator(cell_image)
    anchors = [_on_circle(70), _on_circle(-70)]
    base = _path(calc, anchors)
    # only the left half of the image keeps its speed: the path has to go around the left side
    calc.set_speed_modulation(np.where(np.arange(W) < 80, 1., 0.05)[None].repeat(H, 0))
    assert not np.array_equal(_path(calc, anchors), base)
    calc.set_image(np.repeat(cell_image[..., None], 3, axis=2), sobel_v(cell_image), sobel_h(cell_image))
    calc.set_boundaries(0, 0, W, H)
    assert np.array_equal(_path(calc, anchors), base)


@pytest.mark.parametrize("make_calculator", [_randers_calculator, _symmetric_gradient_calculator])
def test_prior_keeps_the_path_off_the_nucleus_edge(cell_image, make_calculator):
    calc = make_calculator(cell_image)
    anchors = [_on_circle(70), _on_circle(-70)]
    without = _boundary_error(_path(calc, anchors))
    prior = IntensityPrior(strength=0.8)
    prior.set_image(cell_image)
    calc.set_speed_modulation(prior.speed_modulation(*_traced_left_side()))
    with_prior = _boundary_error(_path(calc, anchors))
    assert with_prior < 0.7
    assert with_prior <= without


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


@pytest.fixture(autouse=False)
def restore_prior_state():
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
    yield
    VispyMinimalContourOverlay.set_use_intensity_prior(False)
    VispyMinimalContourOverlay.set_intensity_prior_min_anchors(1)


def _shown(widget):
    # .visible reports whether it is on screen, which it never is with a hidden viewer
    return not widget.native.isHidden()


def test_controls_are_hidden_by_default(viewer, monkeypatch, restore_prior_state):
    mc = _scene(viewer, monkeypatch).mc
    assert not _shown(mc.intensity_prior_checkbox)
    assert not _shown(mc.intensity_prior_strength_slider)
    assert not _shown(mc.intensity_prior_min_anchors_spinbox)
    assert _shown(mc.used_feature_combobox)


@pytest.mark.parametrize("value", ["", "0"])
def test_controls_stay_hidden_for_empty_or_zero(viewer, monkeypatch, restore_prior_state, value):
    mc = _scene(viewer, monkeypatch, env=value).mc
    assert not _shown(mc.intensity_prior_checkbox)


def test_environment_variable_shows_controls_that_are_not_persisted(viewer, monkeypatch, restore_prior_state):
    mc = _scene(viewer, monkeypatch, env="1").mc
    assert _shown(mc.intensity_prior_checkbox)
    assert _shown(mc.intensity_prior_strength_slider)
    assert _shown(mc.intensity_prior_min_anchors_spinbox)
    mc.intensity_prior_checkbox.value = True
    mc.intensity_prior_min_anchors_spinbox.value = 3
    state = mc.__magicclass_serialize__()
    for key in ("intensity_prior_checkbox", "intensity_prior_strength_slider", "intensity_prior_min_anchors_spinbox"):
        assert key not in state
        assert key not in state["Image Features"]


def test_overlay_sets_the_modulation_from_accepted_anchors(viewer, monkeypatch, restore_prior_state):
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
    scene = _scene(viewer, monkeypatch, env="1")
    scene.qt_viewer.controls.widgets[scene.labels].mc_button.click()
    _process_events()
    overlay = scene.labels._overlays["minimal_contour"]
    visual_map = getattr(scene.qt_viewer.canvas, "_layer_overlay_to_visual", None)
    if visual_map is None:
        pytest.skip("this napari version keeps overlay visuals elsewhere")
    vispy_overlay = visual_map[scene.labels][overlay]
    prior = VispyMinimalContourOverlay.intensity_prior
    assert prior.shape == scene.image.data.shape
    # nothing is computed for the prior while it is off
    assert prior._image is None

    calls = []

    class CalculatorSpy:
        def __init__(self, calculator):
            self._calculator = calculator

        def set_speed_modulation(self, modulation):
            calls.append(modulation)
            self._calculator.set_speed_modulation(modulation)

        def __getattr__(self, name):
            return getattr(self._calculator, name)

    monkeypatch.setattr(VispyMinimalContourOverlay, "mc_calculator", CalculatorSpy(VispyMinimalContourOverlay.mc_calculator))
    overlay.anchor_points = [[10., 10.], [10., 40.], [30., 30.]]  # the last one follows the cursor
    overlay.stored_contour = [[10., x] for x in range(10, 41)]

    scene.mc.intensity_prior_checkbox.value = True
    assert calls[-1] is not None and calls[-1].shape == scene.image.data.shape

    overlay.anchor_points = [[10., 10.], [30., 30.]]  # one accepted anchor: enough by default
    overlay.stored_contour = []
    vispy_overlay.update_speed_modulation()
    assert calls[-1] is not None

    scene.mc.intensity_prior_min_anchors_spinbox.value = 2  # recomputes: one anchor is not enough any more
    assert calls[-1] is None
    overlay.anchor_points = [[10., 10.], [10., 40.], [30., 30.]]
    vispy_overlay.update_speed_modulation()
    assert calls[-1] is not None
    scene.mc.intensity_prior_min_anchors_spinbox.value = 3
    assert calls[-1] is None
    scene.mc.intensity_prior_min_anchors_spinbox.value = 1
    assert calls[-1] is not None

    overlay.anchor_points = []  # no anchor at all
    vispy_overlay.update_speed_modulation()
    assert calls[-1] is None
    overlay.anchor_points = [[10., 10.], [10., 40.], [30., 30.]]
    vispy_overlay.update_speed_modulation()
    vispy_overlay.clear(None)
    assert calls[-1] is None

    overlay.anchor_points = [[10., 10.], [10., 40.], [30., 30.]]
    scene.mc.intensity_prior_checkbox.value = False
    assert calls[-1] is None


def test_overlay_samples_the_roi_window_in_window_coordinates(viewer, monkeypatch, restore_prior_state):
    """With a ROI the prior's image is the window; anchors (plane coordinates) are shifted into it."""
    from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import VispyMinimalContourOverlay
    scene = _scene(viewer, monkeypatch, env="1")
    scene.qt_viewer.controls.widgets[scene.labels].mc_button.click()
    _process_events()
    overlay = scene.labels._overlays["minimal_contour"]
    visual_map = getattr(scene.qt_viewer.canvas, "_layer_overlay_to_visual", None)
    if visual_map is None:
        pytest.skip("this napari version keeps overlay visuals elsewhere")
    vispy_overlay = visual_map[scene.labels][overlay]

    # a plane of the scene's shape: dark left of column 45, bright from it; the calculator has columns 30..70
    plane_shape = scene.image.data.shape
    plane = np.where(np.arange(plane_shape[1]) < 45, 0.2, 0.8)[None].repeat(plane_shape[0], 0)
    window = (slice(0, plane_shape[0]), slice(30, plane_shape[1]))
    VispyMinimalContourOverlay.set_prior_image(plane[window])
    VispyMinimalContourOverlay.set_calculator_feature(np.repeat(plane[window][..., None], 3, axis=2),
                                                      roi_origin=(0, 30), plane_shape=plane_shape)

    calls = []
    calculator = VispyMinimalContourOverlay.mc_calculator

    class CalculatorSpy:
        def set_speed_modulation(self, modulation):
            calls.append(modulation)
            calculator.set_speed_modulation(modulation)

        def __getattr__(self, name):
            return getattr(calculator, name)

    monkeypatch.setattr(VispyMinimalContourOverlay, "mc_calculator", CalculatorSpy())
    scene.mc.intensity_prior_checkbox.value = True
    # an anchor on the dark part, at plane column 31 (window column 1), and the cursor; one outside the window
    overlay.anchor_points = [[30., 10.], [30., 31.], [40., 60.]]
    vispy_overlay.update_speed_modulation()
    modulation = calls[-1]
    assert modulation is not None and modulation.shape == plane[window].shape
    # the dark columns keep their speed: the anchor was sampled at the right place (unshifted it would be bright)
    assert modulation[30, 1] > 0.95
    assert modulation[30, 30] < 0.3
    # the anchor outside the window is not sampled: with it alone there is no reference
    overlay.anchor_points = [[30., 10.], [40., 60.]]
    vispy_overlay.update_speed_modulation()
    assert calls[-1] is None
    VispyMinimalContourOverlay.set_calculator_feature(None)
