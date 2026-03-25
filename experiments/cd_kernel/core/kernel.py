"""
Core Christoffel–Darboux kernel reconstruction routines.

This module contains the shared numerical implementation of the CD-kernel
pipeline:
    moments → Toeplitz matrix → kernel diagonal → density proxy.

It is intentionally independent of both dynamical systems and benchmark
measure constructors. Algorithmic variants should call this module rather
than reimplementing the shared linear algebra.
"""

from __future__ import annotations

import numpy as np

from experiments.cd_kernel.core.result import CDKernelResult
from experiments.cd_kernel.core.toeplitz import (
    build_hermitian_toeplitz,
    regularize_toeplitz,
    condition_number,
)
from experiments.cd_kernel.core.normalization import (
    unit_circle_grid,
    normalize_density_proxy,
)

Array = np.ndarray


def christoffel_from_toeplitz(
    T: Array,
    grid_size: int = 2048,
) -> tuple[Array, Array, Array]:
    """
    Evaluate the Christoffel-function proxy
        K(z) = v(z)^* T^{-1} v(z)
    on a grid of unit-circle points.
    """
    T = np.asarray(T, dtype=np.complex128)
    n = T.shape[0] - 1

    angles, z = unit_circle_grid(grid_size)
    kernel_diag = np.zeros(grid_size, dtype=float)

    for idx, zz in enumerate(z):
        v = np.array([zz ** j for j in range(n + 1)], dtype=np.complex128)
        x = np.linalg.solve(T, v)
        kernel_diag[idx] = float(np.real(np.vdot(v, x)))

    return angles, z, kernel_diag


def density_proxy_from_kernel(kernel_diag: Array, floor: float = 1e-14) -> Array:
    kernel_diag = np.asarray(kernel_diag, dtype=float)
    return 1.0 / np.maximum(kernel_diag, floor)

def atomic_mass_proxy_from_kernel(kernel_diag: np.ndarray, order: int, floor: float = 1e-14) -> np.ndarray:
    kernel_diag = np.asarray(kernel_diag, dtype=float)
    invK = 1.0 / np.maximum(kernel_diag, floor)
    raw = invK - 1.0 / (order + 1)
    return np.maximum(raw, 0.0)

def ac_density_proxy_from_kernel(kernel_diag: np.ndarray, order: int, floor: float = 1e-14) -> np.ndarray:
    kernel_diag = np.asarray(kernel_diag, dtype=float)
    invK = 1.0 / np.maximum(kernel_diag, floor)
    raw = (order + 1) * invK - 1.0
    return np.maximum(raw, 0.0)

def modified_moments(moments: np.ndarray) -> np.ndarray:
    moments = np.asarray(moments, dtype=np.complex128).copy()
    moments[0] = moments[0] + 1.0
    return moments


def evaluate_cd_kernel_from_moments(
    moments: Array,
    order: int | None = None,
    grid_size: int = 2048,
    regularization: float = 1e-8,
    normalize_density: bool = True,
) -> CDKernelResult:
    """
    Shared baseline CD-kernel pipeline.

    Parameters
    ----------
    moments:
        Nonnegative moment sequence [m_0, ..., m_M].
    order:
        Toeplitz truncation order. If None, use all available moments.
    grid_size:
        Number of grid points on the unit circle.
    regularization:
        Tikhonov-style diagonal regularization parameter.
    normalize_density:
        Whether to normalize the density proxy to unit integral.
    """
    moments = np.asarray(moments, dtype=np.complex128)
    moments_mod = modified_moments(moments)

    T = build_hermitian_toeplitz(moments_mod, order=order)
    T = regularize_toeplitz(T, regularization)

    order_used = T.shape[0] - 1

    angles, z, kernel_diag = christoffel_from_toeplitz(T, grid_size=grid_size)
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
            "order_used": T.shape[0] - 1,
            "method": "cd_kernel_core",
        },
    )