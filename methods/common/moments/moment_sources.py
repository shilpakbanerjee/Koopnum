"""
Abstractions for generating moments.

This module separates:
- exact moment generation (measure-based)
- empirical moment generation (data/dynamics-based)

This allows us to test the CD-kernel reconstruction independently
of the dynamical system.
"""

from __future__ import annotations
from typing import Protocol
import numpy as np

Array = np.ndarray


class MomentSource(Protocol):
    def moments(self, order: int) -> Array:
        ...


# =========================
# Exact (measure-based)
# =========================

class ExactMomentSource:
    """
    Moment source from a known measure.

    Requires a function:
        m_k = ∫ e^{ikθ} dμ(θ)
    """

    def __init__(self, moment_function):
        self.moment_function = moment_function

    def moments(self, order: int) -> Array:
        return np.array(
            [self.moment_function(k) for k in range(order + 1)],
            dtype=np.complex128,
        )


# =========================
# Empirical (data-based)
# =========================

class EmpiricalMomentSource:
    """
    Moment source from a time series f(x_n).

    Moments are computed as:
        m_k ≈ average of f_n * conj(f_{n+k})
    """

    def __init__(self, signal: Array):
        self.signal = np.asarray(signal, dtype=np.complex128)

    def moments(self, order: int) -> Array:
        n = len(self.signal)
        m = np.zeros(order + 1, dtype=np.complex128)

        for k in range(order + 1):
            m[k] = np.mean(self.signal[: n - k] * np.conj(self.signal[k:]))

        return m