"""
Tapered CD-kernel variant.

This variant modifies the baseline Christoffel–Darboux reconstruction by
replacing the monomial vector

    v(z) = (1, z, z^2, ..., z^N)^T

with a tapered version

    v_w(z) = (w_0, w_1 z, w_2 z^2, ..., w_N z^N)^T,

where the weights w_j downweight the high-order moments. This is a natural
regularization of the truncated moment problem and is closely related to
classical taper/window ideas in Fourier and spectral estimation.

Default taper:
- "fejer"  : w_j = 1 - j/(N+1)
Optional taper:
- "hann"   : w_j = 0.5 * (1 - cos(2*pi*j/N))
- "none"   : recovers the untapered baseline
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from methods.common.results import CDKernelResult
from methods.cd_kernel.core.toeplitz import (
    build_hermitian_toeplitz,
    regularize_toeplitz,
    condition_number,
)
from methods.cd_kernel.core.normalization import (
    unit_circle_grid,
    normalize_density_proxy,
)
from methods.cd_kernel.dynamics.spectral_measure import (
    spectral_measure_data_from_trajectory,
)

Array = np.ndarray
Observable = Callable[[Array], Array]


def taper_weights(order: int, taper: str = "fejer") -> Array:
    if order < 0:
        raise ValueError("order must be nonnegative")

    if taper == "none":
        return np.ones(order + 1, dtype=float)

    j = np.arange(order + 1, dtype=float)

    if taper == "fejer":
        # Cesàro / Fejér weights
        return 1.0 - j / (order + 1)

    if taper == "hann":
        if order == 0:
            return np.ones(1, dtype=float)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * j / order))

    raise ValueError(f"Unknown taper '{taper}'. Use 'fejer', 'hann', or 'none'.")


def tapered_christoffel_from_toeplitz(
    T: Array,
    weights: Array,
    grid_size: int = 2048,
) -> tuple[Array, Array, Array]:
    """
    Evaluate the tapered Christoffel-function proxy

        K_w(z) = v_w(z)^* T^{-1} v_w(z)

    on a grid of unit-circle points.
    """
    T = np.asarray(T, dtype=np.complex128)
    weights = np.asarray(weights, dtype=float)

    n = T.shape[0] - 1
    if weights.shape != (n + 1,):
        raise ValueError("weights must have length n+1, where T is (n+1)x(n+1)")

    angles, z = unit_circle_grid(grid_size)
    kernel_diag = np.zeros(grid_size, dtype=float)

    for idx, zz in enumerate(z):
        v = np.array([weights[j] * (zz ** j) for j in range(n + 1)], dtype=np.complex128)
        x = np.linalg.solve(T, v)
        kernel_diag[idx] = float(np.real(np.vdot(v, x)))

    return angles, z, kernel_diag


def density_proxy_from_kernel(kernel_diag: Array, floor: float = 1e-14) -> Array:
    kernel_diag = np.asarray(kernel_diag, dtype=float)
    return 1.0 / np.maximum(kernel_diag, floor)


def fit_cd_kernel_tapered_from_moments(
    moments: Array,
    order: int | None = None,
    grid_size: int = 2048,
    regularization: float = 1e-8,
    taper: str = "fejer",
    normalize_density: bool = True,
) -> CDKernelResult:
    moments = np.asarray(moments, dtype=np.complex128)

    T = build_hermitian_toeplitz(moments, order=order)
    T = regularize_toeplitz(T, regularization)

    n = T.shape[0] - 1
    weights = taper_weights(n, taper=taper)
    angles, z, kernel_diag = tapered_christoffel_from_toeplitz(
        T,
        weights=weights,
        grid_size=grid_size,
    )
    density_proxy = density_proxy_from_kernel(kernel_diag)

    if normalize_density:
        density_proxy = normalize_density_proxy(density_proxy, angles)

    return CDKernelResult(
        moments=moments,
        toeplitz=T,
        angles=angles,
        circle_points=z,
        kernel_diag=kernel_diag,
        density_proxy=density_proxy,
        regularization=regularization,
        metadata={
            "grid_size": grid_size,
            "toeplitz_condition_number": condition_number(T),
            "order_used": n,
            "method": "cd_kernel_tapered",
            "taper": taper,
            "weights": weights,
        },
    )


def fit_cd_kernel_tapered(
    X: Array,
    order: int,
    observable: Optional[Observable] = None,
    grid_size: int = 2048,
    regularization: float = 1e-8,
    taper: str = "fejer",
    center: bool = True,
    normalize_moments: bool = True,
    taper_signal: Optional[Array] = None,
    normalize_density: bool = True,
) -> CDKernelResult:
    """
    Tapered CD-kernel reconstruction from a trajectory.

    Pipeline:
        trajectory → observable signal → empirical moments → tapered CD reconstruction
    """
    spec = spectral_measure_data_from_trajectory(
        X,
        order=order,
        observable=observable,
        center=center,
        normalize=normalize_moments,
        taper=taper_signal,
    )

    result = fit_cd_kernel_tapered_from_moments(
        moments=spec.moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        taper=taper,
        normalize_density=normalize_density,
    )

    result.metadata["variant"] = "cd_kernel_v002_tapered"
    result.metadata["signal_length"] = len(spec.signal)
    return result