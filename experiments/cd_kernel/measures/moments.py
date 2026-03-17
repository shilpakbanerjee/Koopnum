"""
Moment computation utilities for measures on the unit circle.

This module implements routines for constructing moment sequences
    m_k = ∫ z^k dμ(z)
either from:
    (i) time-series data (via correlation estimates), or
    (ii) explicit descriptions of measures (atomic or absolutely continuous).

These moments form the fundamental input to the truncated moment problem
and subsequent Christoffel–Darboux (CD) kernel reconstruction.

The functionality here is completely independent of dynamical systems:
it operates purely at the level of measures and their moments.

Key features:
- Empirical moment estimation from signals (with optional tapering)
- Exact moments for atomic measures
- Numerical quadrature for absolutely continuous measures
- Optional normalization (m_0 = 1)

This file constitutes the first stage of the pipeline:
    measure → moments
"""


from __future__ import annotations

from typing import Callable, Optional
import numpy as np

Array = np.ndarray


def ensure_complex_1d(signal: Array) -> Array:
    x = np.asarray(signal)
    if x.ndim != 1:
        raise ValueError("signal must be a 1D array")
    return x.astype(np.complex128, copy=False)


def normalize_moments(moments: Array, atol: float = 1e-15) -> Array:
    moments = np.asarray(moments, dtype=np.complex128)
    if abs(moments[0]) <= atol:
        return moments.copy()
    return moments / moments[0]


def estimate_moments_from_signal(
    signal: Array,
    order: int,
    center: bool = True,
    normalize: bool = True,
    taper: Optional[Array] = None,
) -> Array:
    """
    Estimate moments m_k = <U^k f, f> from a scalar time series.

    Parameters
    ----------
    signal:
        1D array of observable values along a trajectory.
    order:
        Maximum lag/moment order.
    center:
        If True, subtract the mean from the signal.
    normalize:
        If True, divide moments by m_0.
    taper:
        Optional 1D taper/window of same length as signal.

    Returns
    -------
    moments:
        Array [m_0, ..., m_order].
    """
    x = ensure_complex_1d(signal)
    n = x.size
    if order >= n:
        raise ValueError("order must be smaller than signal length")

    if center:
        x = x - np.mean(x)

    if taper is not None:
        taper = np.asarray(taper, dtype=float)
        if taper.shape != x.shape:
            raise ValueError("taper must have same shape as signal")
        x = taper * x

    moments = np.zeros(order + 1, dtype=np.complex128)
    for k in range(order + 1):
        moments[k] = np.vdot(x[: n - k], x[k:]) / (n - k)

    if normalize:
        moments = normalize_moments(moments)

    return moments


def moments_from_atomic_measure(
    angles: Array,
    weights: Array,
    order: int,
    normalize: bool = True,
) -> Array:
    """
    Moments of an atomic measure:
        mu = sum_j w_j delta_{exp(i theta_j)}

    m_k = int z^k dmu(z) = sum_j w_j exp(i k theta_j)
    """
    angles = np.asarray(angles, dtype=float)
    weights = np.asarray(weights, dtype=np.complex128)

    if angles.ndim != 1 or weights.ndim != 1:
        raise ValueError("angles and weights must be 1D")
    if angles.size != weights.size:
        raise ValueError("angles and weights must have same length")

    moments = np.zeros(order + 1, dtype=np.complex128)
    z = np.exp(1j * angles)
    for k in range(order + 1):
        moments[k] = np.sum(weights * (z ** k))

    if normalize:
        moments = normalize_moments(moments)

    return moments


def moments_from_density(
    density_fn: Callable[[Array], Array],
    order: int,
    grid_size: int = 4096,
    normalize: bool = True,
) -> Array:
    """
    Approximate moments of an absolutely continuous measure
        dmu(theta) = rho(theta) dtheta
    on [0, 2pi), using numerical quadrature.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
    rho = np.asarray(density_fn(angles), dtype=np.complex128)

    if rho.shape != angles.shape:
        raise ValueError("density_fn must return array of same shape as angles")

    moments = np.zeros(order + 1, dtype=np.complex128)
    z = np.exp(-1j * angles)

    if hasattr(np, "trapezoid"):
        trapz = np.trapezoid
    else:
        trapz = np.trapz

    for k in range(order + 1):
        integrand = (z ** k) * rho
        moments[k] = trapz(integrand, angles)

    if normalize:
        moments = normalize_moments(moments)

    return moments


def hann_taper(n: int) -> Array:
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=float)
    j = np.arange(n)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * j / (n - 1)))


def hamming_taper(n: int) -> Array:
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=float)
    j = np.arange(n)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * j / (n - 1))