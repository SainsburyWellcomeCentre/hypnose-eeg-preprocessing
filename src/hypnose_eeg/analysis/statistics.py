"""Reusable robust statistical calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def robust_upper_z(values: ArrayLike, mad_scale: float = 1.4826) -> np.ndarray:
    """Return log-scaled robust z-scores for detecting unusually high values.

    Finite values are transformed with ``log10`` and standardized using their
    median and median absolute deviation (MAD). Non-finite values remain NaN.
    """
    if not np.isfinite(mad_scale) or mad_scale <= 0:
        raise ValueError("mad_scale must be a positive finite number")

    array = np.asarray(values, dtype=float)
    transformed = np.log10(np.clip(array, np.finfo(float).tiny, None))
    finite = np.isfinite(transformed)
    result = np.full(transformed.shape, np.nan)
    if not finite.any():
        return result

    median = np.median(transformed[finite])
    mad = np.median(np.abs(transformed[finite] - median))
    result[finite] = (
        0.0 if mad == 0 else (transformed[finite] - median) / (mad_scale * mad)
    )
    return result
