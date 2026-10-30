"""Intensity prior for the minimal contour (experimental).

The boundary of an object often has roughly the same intensity all around. This module estimates that
intensity from the part of the boundary the user has already traced (the anchor points and the accepted
contour) and turns it into a per-pixel speed factor for the minimal path calculation: pixels with a
matching intensity keep their speed, others are slowed down. The factor is never above 1, so the prior
can only make paths more expensive. When the traced boundary does not have a consistent intensity
(e.g. DIC images) the estimated spread grows and the factor approaches 1 everywhere, so the result is
the same as without the prior.

The factor is exp(-ln(max_slowdown) * (1 - S)), where S in [0, 1] is the similarity of a pixel to the
reference intensity: matching pixels keep their speed, the most different ones move max_slowdown times
slower. The image features themselves make edge pixels hundreds of times faster than others, so the
slowdown has to be of that order to change which edge the path follows.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# scales the median absolute deviation to the standard deviation of a normal distribution
MAD_TO_SIGMA = 1.4826
# the calculator clamps the speed factor to 1e-3
MAX_SLOWDOWN = 1000.
# the tolerance is at least this many times the noise of the image ...
NOISE_FACTOR = 3.
# ... and never below this (for noise free images; intensities are expected in about [0, 1])
MIN_SIGMA_FLOOR = 1e-3
# sigma of the Gaussian whose residual the noise is estimated from
NOISE_HIGH_PASS_SIGMA = 3.
# the tolerance is at least the intensity change over this many pixels across the traced boundary
SLOPE_FACTOR = 1.
# what the contour follows (see IntensityPrior)
EDGE = "edge"
BRIGHT_LINE = "bright line"
DARK_LINE = "dark line"
BOUNDARY_TYPES = (EDGE, BRIGHT_LINE, DARK_LINE)


class IntensityPrior:
    """Estimates the boundary intensity from traced points and computes the speed modulation.

    Parameters
    ----------
    max_slowdown : float
        How many times slower the pixels with the most different intensity get, in [1, 1000]. 1 switches the
        prior off.
    min_sigma : float or None
        Lower limit of the intensity tolerance. None (the default) derives it from the image: NOISE_FACTOR
        times its noise, so the tolerance follows the image's own scale (a dim membrane stain and a bright
        nuclear stain get very different tolerances after scaling to [0, 1]).
    recent_length : int
        Number of the most recently accepted contour pixels used as samples, so the reference follows slow
        intensity changes along the boundary.
    snap_radius : int
        Anchor points are sampled at the pixel within this radius that is most typical of the boundary type: a
        click a pixel off the boundary would otherwise sample the inside or outside of the object.
    smoothing : float
        Sigma of the Gaussian applied to the image before sampling and before computing the similarity.
    min_anchors : int
        Number of accepted anchor points needed before the prior is used (at least 1). With a single anchor the
        reference is that one sample and the tolerance is the lower limit; from the second anchor on the accepted
        contour adds samples and the tolerance is estimated from them.
    boundary : str
        What the contour follows, one of BOUNDARY_TYPES:

        - "edge": the border between a brighter and a darker region (e.g. a stained nucleus). The contour stays
          on the intensity level of the traced part; anchors are sampled at the strongest gradient nearby.
        - "bright line": a bright ridge (e.g. a membrane stain). Pixels brighter than the reference also match:
          a ridge's brightness varies, and where it is brighter than the reference the pixels with the reference
          intensity are on its flanks, where a two-sided similarity would pull the contour. Anchors are sampled
          at the brightest pixel nearby.
        - "dark line": the same for a dark valley.
    """

    def __init__(self, max_slowdown=100., min_sigma=None, recent_length=100, snap_radius=2, smoothing=1.,
                 min_anchors=1, boundary=EDGE):
        self.max_slowdown = max_slowdown
        self.min_anchors = min_anchors
        self.boundary = boundary
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
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, value):
        if value not in BOUNDARY_TYPES:
            raise ValueError("boundary should be one of %s, got %r" % (BOUNDARY_TYPES, value))
        self._boundary = value

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

    def _snap_to_boundary(self, point):
        r = self.snap_radius
        row, col = point
        if r <= 0:
            return row, col
        r0, c0 = max(row - r, 0), max(col - r, 0)
        window = (slice(r0, row + r + 1), slice(c0, col + r + 1))
        if self.boundary == EDGE:
            score = self.grad_magnitude[window]
        elif self.boundary == BRIGHT_LINE:
            score = self.image[window]
        else:
            score = -self.image[window]
        d_row, d_col = np.unravel_index(np.argmax(score), score.shape)
        return r0 + d_row, c0 + d_col

    def _sample_indices(self, anchors, contour=None):
        """(rows, columns) of the (snapped) anchors and the most recent accepted contour pixels."""
        indices = [self._snap_to_boundary(p) for p in self._pixel_indices(anchors)]
        if contour is not None and len(contour) and self.recent_length > 0:
            indices.extend(map(tuple, self._pixel_indices(contour)[-self.recent_length:]))
        indices = np.asarray(indices, dtype=int).reshape(-1, 2)
        return indices[:, 0], indices[:, 1]

    def samples(self, anchors, contour=None):
        """Intensities sampled at the (snapped) anchors and the most recent accepted contour pixels.

        `anchors` and `contour` are (N, 2) arrays of (row, column) image coordinates; `contour` is ordered
        from the oldest to the newest pixel.
        """
        if self._raw_image is None:
            return np.empty(0)
        return self.image[self._sample_indices(anchors, contour)]

    def reference(self, anchors, contour=None):
        """(mu, sigma) of the boundary intensity, or None while fewer than ``min_anchors`` anchors are accepted.

        sigma is the spread of the samples, but at least the larger of ``sigma_floor`` and the intensity change
        over one pixel across the boundary (SLOPE_FACTOR times the median gradient at the samples): on a steep
        edge no pixel may have exactly the reference intensity, and the pixels on the boundary must still match.
        """
        if self._raw_image is None or len(anchors) < self.min_anchors:
            return None
        indices = self._sample_indices(anchors, contour)
        sampled = self.image[indices]
        mu = float(np.median(sampled))
        sigma = MAD_TO_SIGMA * float(np.median(np.abs(sampled - mu)))
        slope = SLOPE_FACTOR * float(np.median(self.grad_magnitude[indices]))
        return mu, max(sigma, self.sigma_floor, slope)

    def speed_modulation(self, anchors, contour=None):
        """Per-pixel speed factor in [1 / max_slowdown, 1] for the whole image, or None if the prior is inactive."""
        if self.max_slowdown <= 1:
            return None
        ref = self.reference(anchors, contour)
        if ref is None:
            return None
        mu, sigma = ref
        difference = self.image - mu
        if self.boundary == BRIGHT_LINE:
            difference = np.minimum(difference, 0.)
        elif self.boundary == DARK_LINE:
            difference = np.maximum(difference, 0.)
        similarity = np.exp(-0.5 * (difference / sigma) ** 2)
        return np.exp(np.log(self.max_slowdown) * (similarity - 1.))
