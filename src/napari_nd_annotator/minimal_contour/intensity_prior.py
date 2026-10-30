"""Intensity prior for the minimal contour (experimental).

The boundary of an object often has roughly the same intensity all around. This module estimates that
intensity from the part of the boundary the user has already traced (the anchor points and the accepted
contour) and turns it into a per-pixel speed factor for the minimal path calculation: pixels with a
matching intensity keep their speed, others are slowed down. The factor is never above 1, so the prior
can only make paths more expensive. When the traced boundary does not have a consistent intensity
(e.g. DIC images) the estimated spread grows and the factor approaches 1 everywhere, so the result is
the same as without the prior.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# scales the median absolute deviation to the standard deviation of a normal distribution
MAD_TO_SIGMA = 1.4826
MAX_STRENGTH = 0.95


class IntensityPrior:
    """Estimates the boundary intensity from traced points and computes the speed modulation.

    Parameters
    ----------
    strength : float
        How much pixels with a different intensity are slowed down, in [0, 0.95]. 0 switches the prior off;
        with 0.8 the most different pixels move at 20% of their original speed.
    min_sigma : float
        Lower limit of the intensity tolerance (the image is expected to be scaled to about [0, 1]).
    recent_length : int
        Number of the most recently accepted contour pixels used as samples, so the reference follows slow
        intensity changes along the boundary.
    snap_radius : int
        Anchor points are sampled at the pixel with the highest gradient within this radius: a click a pixel
        off the edge would otherwise sample the inside or outside of the object.
    smoothing : float
        Sigma of the Gaussian applied to the image before sampling and before computing the similarity.
    min_anchors : int
        Number of accepted anchor points needed before the prior is used (at least 1). With a single anchor the
        reference is that one sample and the tolerance is ``min_sigma``; from the second anchor on the accepted
        contour adds samples and the tolerance is estimated from them.
    """

    def __init__(self, strength=0.8, min_sigma=0.05, recent_length=100, snap_radius=2, smoothing=1., min_anchors=1):
        self.strength = strength
        self.min_anchors = min_anchors
        self.min_sigma = min_sigma
        self.recent_length = recent_length
        self.snap_radius = snap_radius
        self.smoothing = smoothing
        # the image as given; it is only smoothed when the prior is first used on it
        self._raw_image = None
        self._image = None
        self._grad_magnitude = None

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, value):
        self._strength = float(np.clip(value, 0., MAX_STRENGTH))

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

    @property
    def grad_magnitude(self):
        if self._grad_magnitude is None and self.image is not None:
            self._grad_magnitude = np.hypot(*np.gradient(self.image))
        return self._grad_magnitude

    def _pixel_indices(self, points):
        points = np.round(np.asarray(points, dtype=float).reshape(-1, 2)).astype(int)
        return np.clip(points, 0, np.subtract(self.shape, 1))

    def _snap_to_edge(self, point):
        r = self.snap_radius
        row, col = point
        if r <= 0:
            return row, col
        r0, c0 = max(row - r, 0), max(col - r, 0)
        window = self.grad_magnitude[r0:row + r + 1, c0:col + r + 1]
        d_row, d_col = np.unravel_index(np.argmax(window), window.shape)
        return r0 + d_row, c0 + d_col

    def samples(self, anchors, contour=None):
        """Intensities sampled at the (snapped) anchors and the most recent accepted contour pixels.

        `anchors` and `contour` are (N, 2) arrays of (row, column) image coordinates; `contour` is ordered
        from the oldest to the newest pixel.
        """
        if self._raw_image is None:
            return np.empty(0)
        anchor_idx = [self._snap_to_edge(p) for p in self._pixel_indices(anchors)]
        sampled = [self.image[r, c] for r, c in anchor_idx]
        if contour is not None and len(contour) and self.recent_length > 0:
            contour_idx = self._pixel_indices(contour)[-self.recent_length:]
            sampled.extend(self.image[contour_idx[:, 0], contour_idx[:, 1]])
        return np.asarray(sampled, dtype=float)

    def reference(self, anchors, contour=None):
        """(mu, sigma) of the boundary intensity, or None while fewer than ``min_anchors`` anchors are accepted."""
        if self._raw_image is None or len(anchors) < self.min_anchors:
            return None
        sampled = self.samples(anchors, contour)
        mu = float(np.median(sampled))
        sigma = MAD_TO_SIGMA * float(np.median(np.abs(sampled - mu)))
        return mu, max(sigma, self.min_sigma)

    def speed_modulation(self, anchors, contour=None):
        """Per-pixel speed factor in [1 - strength, 1] for the whole image, or None if the prior is inactive."""
        if self.strength <= 0:
            return None
        ref = self.reference(anchors, contour)
        if ref is None:
            return None
        mu, sigma = ref
        similarity = np.exp(-0.5 * ((self.image - mu) / sigma) ** 2)
        return (1. - self.strength) + self.strength * similarity
