"""Shared velocity limits and reflecting boundaries for PSO variants."""

import numpy as np


def velocity_limit(lower, upper, v_max=None):
    """Default to 20% of the range; explicit limits are absolute magnitudes."""
    limit = 0.2 * (upper - lower) if v_max is None else v_max
    limit = np.broadcast_to(np.asarray(limit, dtype=float), lower.shape).copy()
    if np.any(limit < 0) or np.any(np.isnan(limit)):
        raise ValueError("v_max must be nonnegative")
    return limit


def reflect_bounds(positions, velocities, lower, upper):
    """Fold arbitrary overshoots into bounds, reversing velocity at each bounce.

    Handles multiple crossings without loops. Fixed dimensions are pinned with
    zero velocity. Both input arrays are updated in place.
    """
    width = upper - lower
    fixed = width == 0
    safe_width = np.where(fixed, 1.0, width)
    outside = (positions < lower) | (positions > upper)
    phase = np.remainder(positions - lower, 2 * safe_width)
    folded = lower + np.minimum(phase, 2 * safe_width - phase)
    positions[:] = np.where(outside, folded, positions)
    velocities[outside & (phase >= safe_width)] *= -1
    positions[:, fixed] = lower[fixed]
    velocities[:, fixed] = 0
