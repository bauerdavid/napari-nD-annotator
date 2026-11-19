"""Intensity prior for the minimal contour (experimental).

The boundary of an object often has roughly the same intensity all around. This module takes that intensity
from where the user clicked (the anchor points, and the contour accepted between them) and makes the contour
follow pixels of similar intensity: while the prior is active, the speed of the minimal path calculation is

    speed = intensity_speed * feature_speed ** feature_weight

``intensity_speed`` = exp(-ln(max_slowdown) * (1 - S)), where S in [0, 1] is the similarity of a pixel's
intensity to the reference: matching pixels get full speed, the most different ones max_slowdown times less.
``feature_speed`` is the speed the selected image feature (gradient, intensity, custom) gives the pixel, scaled
to [0, 1]; with a small ``feature_weight`` it only breaks ties and smooths out noise. With 0 the contour follows
the intensity alone. This speed replaces the metric of the selected feature (see
MinimalContourCalculator.set_speed_map).

When the traced boundary does not have a consistent intensity (e.g. DIC images), the estimated spread grows
and the similarity approaches 1 everywhere, so the contour is driven by the feature term alone.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# scales the median absolute deviation to the standard deviation of a normal distribution
MAD_TO_SIGMA = 1.4826
MAX_SLOWDOWN = 1000.
# the tolerance is at least this many times the noise of the image ...
NOISE_FACTOR = 3.
# ... and never below this (for noise free images; intensities are expected in about [0, 1])
MIN_SIGMA_FLOOR = 1e-3
# sigma of the Gaussian whose residual the noise is estimated from
NOISE_HIGH_PASS_SIGMA = 3.
# the tolerance is at least the intensity change over this many pixels at the sampled points
SLOPE_FACTOR = 1.
# the feature speed is at least this, so that a pixel without any feature can still be crossed
FEATURE_SPEED_FLOOR = 1e-3
# which intensities match the reference (see IntensityPrior)
SIMILAR = "similar"
SIMILAR_OR_BRIGHTER = "similar or brighter"
SIMILAR_OR_DARKER = "similar or darker"
MATCH_TYPES = (SIMILAR, SIMILAR_OR_BRIGHTER, SIMILAR_OR_DARKER)


def feature_speed(features, grad_x=None, grad_y=None):
    """The speed the selected image feature gives each pixel, scaled to [0, 1].

    ``features``, ``grad_x`` and ``grad_y`` are what the calculator gets: with gradients (the asymmetric gradient
    feature) it is the gradient magnitude, otherwise the (first channel of the) features, which are already
    "higher is faster" (symmetric gradient, high or low intensity, custom). The 1st and 99th percentiles are mapped
    to 0 and 1, so a few extreme pixels do not compress the rest.
    """
    if grad_x is not None and np.size(grad_x):
        speed = np.hypot(grad_x, grad_y)
    else:
        speed = np.asarray(features, dtype=float)
        if speed.ndim == 3:
            speed = speed[..., 0]
    low, high = np.percentile(speed, (1, 99))
    if not high > low:
        return np.ones(speed.shape)
    return np.clip((speed - low) / (high - low), 0., 1.)


class IntensityPrior:
    """Takes the reference intensity from the anchors and computes the speed map of the contour.

    Parameters
    ----------
    max_slowdown : float
        How many times slower the pixels with the most different intensity get, in [1, 1000]. 1 switches the
        prior off.
    feature_weight : float
        Exponent of the feature speed in the speed map, in [0, 1]. 0: the intensity alone drives the contour.
    min_sigma : float or None
        Lower limit of the intensity tolerance. None (the default) derives it from the image: NOISE_FACTOR
        times its noise, so the tolerance follows the image's own scale (a dim membrane stain and a bright
        nuclear stain get very different tolerances after scaling to [0, 1]).
    recent_length : int
        Number of the most recently accepted contour pixels used as samples, so the reference follows slow
        intensity changes along the boundary.
    snap_radius : int
        With a positive radius, an anchor is not sampled where it was clicked but at the pixel within this radius
        that is most typical of the match type: the strongest gradient for "similar", the brightest pixel for
        "similar or brighter" and the darkest one for "similar or darker". 0 (the default) samples the clicked
        pixel.
    smoothing : float
        Sigma of the Gaussian applied to the image before sampling and before computing the similarity.
    min_anchors : int
        Number of accepted anchor points needed before the prior is used (at least 1). With a single anchor the
        reference is that one sample and the tolerance is the lower limit; from the second anchor on the accepted
        contour adds samples and the tolerance is estimated from them.
    match : str
        Which intensities match the reference, one of MATCH_TYPES:

        - "similar": intensities close to the reference, darker or brighter.
        - "similar or brighter": also anything brighter. For bright lines such as a membrane stain: their
          brightness varies, and where a line is brighter than the reference the pixels with the reference
          intensity are on its flanks, where a two-sided similarity would pull the contour.
        - "similar or darker": the same for dark lines.
    """

    def __init__(self, max_slowdown=100., feature_weight=0.25, min_sigma=None, recent_length=100, snap_radius=0,
                 smoothing=1., min_anchors=1, match=SIMILAR):
        self.max_slowdown = max_slowdown
        self.feature_weight = feature_weight
        self.min_anchors = min_anchors
        self.match = match
        self.min_sigma = min_sigma
        self.recent_length = recent_length
        self.snap_radius = snap_radius
        self.smoothing = smoothing
        # the image as given; it is only smoothed when the prior is first used on it
        self._raw_image = None
        self._image = None
        self._grad_magnitude = None
        self._noise_sigma = None

    @property
    def max_slowdown(self):
        return self._max_slowdown

    @max_slowdown.setter
    def max_slowdown(self, value):
        self._max_slowdown = float(np.clip(value, 1., MAX_SLOWDOWN))

    @property
    def feature_weight(self):
        return self._feature_weight

    @feature_weight.setter
    def feature_weight(self, value):
        self._feature_weight = float(np.clip(value, 0., 1.))

    @property
    def match(self):
        return self._match

    @match.setter
    def match(self, value):
        if value not in MATCH_TYPES:
            raise ValueError("match should be one of %s, got %r" % (MATCH_TYPES, value))
        self._match = value

    @property
    def min_anchors(self):
        return self._min_anchors

    @min_anchors.setter
    def min_anchors(self, value):
        # without an anchor there is nothing to sample
        self._min_anchors = max(int(value), 1)

    @property
    def shape(self):
        """Shape of the current image, or None if there is no image."""
        return None if self._raw_image is None else self._raw_image.shape[:2]

    @property
    def image(self):
        """The smoothed, single channel image the intensities are read from."""
        if self._image is None and self._raw_image is not None:
            image = np.asarray(self._raw_image, dtype=float)
            if image.ndim == 3:
                image = image.mean(-1)
            if self.smoothing > 0:
                image = gaussian_filter(image, self.smoothing)
            self._image = image
        return self._image

    def set_image(self, image):
        """Set the 2D image (or 2D + channels, which are averaged) the intensities are read from."""
        if image is not None and np.ndim(image) not in (2, 3):
            raise ValueError("image should be 2D (optionally with channels), got shape %s" % (np.shape(image),))
        self._raw_image = image
        self._image = None
        self._grad_magnitude = None
        self._noise_sigma = None

    @property
    def noise_sigma(self):
        """Robust estimate of the noise of `image`: the spread of what a Gaussian blur removes from it."""
        if self._noise_sigma is None and self.image is not None:
            residual = self.image - gaussian_filter(self.image, NOISE_HIGH_PASS_SIGMA)
            self._noise_sigma = MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
        return self._noise_sigma

    @property
    def sigma_floor(self):
        """The lower limit of the intensity tolerance (None without an image)."""
        if self.min_sigma is not None:
            return self.min_sigma
        if self.image is None:
            return None
        return max(NOISE_FACTOR * self.noise_sigma, MIN_SIGMA_FLOOR)

    @property
    def grad_magnitude(self):
        if self._grad_magnitude is None and self.image is not None:
            self._grad_magnitude = np.hypot(*np.gradient(self.image))
        return self._grad_magnitude

    def _pixel_indices(self, points):
        points = np.round(np.asarray(points, dtype=float).reshape(-1, 2)).astype(int)
        return np.clip(points, 0, np.subtract(self.shape, 1))

    def _snap(self, point):
        r = self.snap_radius
        row, col = point
        if r <= 0:
            return row, col
        r0, c0 = max(row - r, 0), max(col - r, 0)
        window = (slice(r0, row + r + 1), slice(c0, col + r + 1))
        if self.match == SIMILAR:
            score = self.grad_magnitude[window]
        elif self.match == SIMILAR_OR_BRIGHTER:
            score = self.image[window]
        else:
            score = -self.image[window]
        d_row, d_col = np.unravel_index(np.argmax(score), score.shape)
        return r0 + d_row, c0 + d_col

    def _sample_indices(self, anchors, contour=None):
        """(rows, columns) of the anchors (snapped, see snap_radius) and the most recent accepted contour pixels."""
        indices = [self._snap(p) for p in self._pixel_indices(anchors)]
        if contour is not None and len(contour) and self.recent_length > 0:
            indices.extend(map(tuple, self._pixel_indices(contour)[-self.recent_length:]))
        indices = np.asarray(indices, dtype=int).reshape(-1, 2)
        return indices[:, 0], indices[:, 1]

    def samples(self, anchors, contour=None):
        """Intensities sampled at the anchors and the most recent accepted contour pixels.

        `anchors` and `contour` are (N, 2) arrays of (row, column) image coordinates; `contour` is ordered
        from the oldest to the newest pixel.
        """
        if self._raw_image is None:
            return np.empty(0)
        return self.image[self._sample_indices(anchors, contour)]

    def reference(self, anchors, contour=None):
        """(mu, sigma) of the reference intensity, or None while fewer than ``min_anchors`` anchors are accepted.

        sigma is the spread of the samples, but at least the larger of ``sigma_floor`` and the intensity change
        over one pixel at the samples (SLOPE_FACTOR times their median gradient): where the intensity changes
        quickly, no pixel may have exactly the reference intensity, and the line of that intensity must still be
        followed.
        """
        if self._raw_image is None or len(anchors) < self.min_anchors:
            return None
        indices = self._sample_indices(anchors, contour)
        sampled = self.image[indices]
        mu = float(np.median(sampled))
        sigma = MAD_TO_SIGMA * float(np.median(np.abs(sampled - mu)))
        slope = SLOPE_FACTOR * float(np.median(self.grad_magnitude[indices]))
        return mu, max(sigma, self.sigma_floor, slope)

    def intensity_speed(self, anchors, contour=None):
        """Per-pixel speed in [1 / max_slowdown, 1] from the intensity alone, or None if the prior is inactive."""
        if self.max_slowdown <= 1:
            return None
        ref = self.reference(anchors, contour)
        if ref is None:
            return None
        mu, sigma = ref
        difference = self.image - mu
        if self.match == SIMILAR_OR_BRIGHTER:
            difference = np.minimum(difference, 0.)
        elif self.match == SIMILAR_OR_DARKER:
            difference = np.maximum(difference, 0.)
        similarity = np.exp(-0.5 * (difference / sigma) ** 2)
        return np.exp(np.log(self.max_slowdown) * (similarity - 1.))

    def speed_map(self, anchors, contour=None, feature=None):
        """The speed of the contour, or None if the prior is inactive.

        ``feature`` is the feature speed (see `feature_speed`) of the same image, or None to use the intensity
        alone.
        """
        speed = self.intensity_speed(anchors, contour)
        if speed is None or feature is None or self.feature_weight <= 0:
            return speed
        return speed * np.maximum(feature, FEATURE_SPEED_FLOOR) ** self.feature_weight
