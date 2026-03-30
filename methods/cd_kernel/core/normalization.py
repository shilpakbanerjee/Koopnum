"""
Normalization utilities for CD-kernel reconstruction.

This module provides unit-circle quadrature and normalization helpers for
density proxies produced by the Christoffel–Darboux pipeline.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def circle_integral(values: Array, angles: Array) -> float:
    values = np.asarray(values)
    angles = np.asarray(angles, dtype=float)
    if hasattr(np, "trapezoid"):
        val = np.trapezoid(values, angles)
    else:
        val = np.trapz(values, angles)
    return float(np.real(val))


def normalize_density_proxy(density: Array, angles: Array) -> Array:
    density = np.asarray(density, dtype=float)
    integral = circle_integral(density, angles)
    if integral <= 0:
        return density.copy()
    return density / integral


def unit_circle_grid(grid_size: int) -> tuple[Array, Array]:
    angles = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
    z = np.exp(1j * angles)
    return angles, z