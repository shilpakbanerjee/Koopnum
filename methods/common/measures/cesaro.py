from __future__ import annotations

from dataclasses import dataclass
import numpy as np

Array = np.ndarray


@dataclass
class CesaroResult:
    """
    Cesàro / Fejér weak reconstruction result from a truncated moment sequence.
    """
    moments: Array
    order: int
    angles: Array
    density: Array
    cdf: Array
    metadata: dict


def _validate_moments(moments: Array, order: int | None = None) -> tuple[Array, int]:
    moments = np.asarray(moments, dtype=np.complex128)
    if moments.ndim != 1:
        raise ValueError("moments must be a 1D array")
    if len(moments) == 0:
        raise ValueError("moments must be nonempty")

    max_order = len(moments) - 1
    if order is None:
        order = max_order

    if order < 0 or order > max_order:
        raise ValueError(f"order must satisfy 0 <= order <= {max_order}")

    return moments, int(order)


def fourier_partial_sum(moments: Array, n: int, angles: Array) -> Array:
    """
    Compute the n-th symmetric Fourier partial sum
        S_n(theta) = sum_{k=-n}^n m_{-k} e^{ik theta},
    using the Hermitian relation m_{-k} = conjugate(m_k).
    """
    moments = np.asarray(moments, dtype=np.complex128)
    angles = np.asarray(angles, dtype=float)

    if n < 0 or n >= len(moments):
        raise ValueError("n must satisfy 0 <= n <= len(moments)-1")

    out = np.full_like(angles, np.real(moments[0]), dtype=np.complex128)

    for k in range(1, n + 1):
        out += np.conjugate(moments[k]) * np.exp(1j * k * angles)
        out += moments[k] * np.exp(-1j * k * angles)

    return out


def cumulative_distribution_from_density(density: Array, angles: Array) -> Array:
    """
    Build a discrete right-continuous CDF from density values on a uniform angular grid.
    """
    density = np.asarray(density, dtype=float)
    angles = np.asarray(angles, dtype=float)

    if density.shape != angles.shape:
        raise ValueError("density and angles must have the same shape")

    if len(angles) < 2:
        return np.array([0.0], dtype=float)

    dtheta = float(angles[1] - angles[0])
    return np.cumsum(density) * dtheta


def _trapz_periodic(values: Array, angles: Array) -> float:
    values = np.asarray(values, dtype=float)
    angles = np.asarray(angles, dtype=float)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, angles))
    return float(np.trapz(values, angles))


def cesaro_density_from_moments(
    moments: Array,
    order: int | None = None,
    grid_size: int = 2048,
    clip_negative: bool = True,
    normalize_mass: bool = True,
) -> CesaroResult:
    """
    Compute the Cesàro / Fejér density approximation from a truncated moment sequence.

    Parameters
    ----------
    moments:
        Array [m_0, ..., m_M].
    order:
        Reconstruction order N. If None, use all available moments.
    grid_size:
        Number of angular grid points in [0, 2pi).
    clip_negative:
        Clip small negative numerical values to zero.
    normalize_mass:
        Renormalize density to total mass Re(m_0).

    Returns
    -------
    CesaroResult
    """
    moments, order = _validate_moments(moments, order=order)

    angles = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
    rho = np.zeros(grid_size, dtype=np.complex128)

    for n in range(order + 1):
        rho += fourier_partial_sum(moments, n=n, angles=angles)

    rho /= (order + 1)
    rho = np.real(rho)

    if clip_negative:
        rho = np.maximum(rho, 0.0)

    total_mass_target = float(np.real(moments[0]))

    if normalize_mass and total_mass_target > 0:
        current_mass = _trapz_periodic(rho, angles)
        if current_mass > 0:
            rho = rho * (total_mass_target / current_mass)

    cdf = cumulative_distribution_from_density(rho, angles)

    return CesaroResult(
        moments=moments,
        order=order,
        angles=angles,
        density=rho,
        cdf=cdf,
        metadata={
            "method": "cesaro_fejer",
            "grid_size": grid_size,
            "total_mass_target": total_mass_target,
            "clip_negative": clip_negative,
            "normalize_mass": normalize_mass,
        },
    )