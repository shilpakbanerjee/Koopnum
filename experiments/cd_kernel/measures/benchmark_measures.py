"""
Benchmark measures for validating CD-kernel reconstruction.

This module defines simple classes and constructors for measures on the
unit circle with known analytical structure, enabling independent testing
of the numerical reconstruction pipeline.

Supported measure types:
- Atomic measures:
      μ = Σ w_j δ_{e^{iθ_j}}
- Absolutely continuous measures:
      dμ(θ) = ρ(θ) dθ

It also includes common example densities such as:
- uniform (Lebesgue) measure
- cosine-modulated densities
- wrapped Gaussian-type densities

Each measure provides a method to generate its moment sequence, allowing
direct comparison between:
    exact measure ↔ reconstructed measure

This module is used for validation of the measure-reconstruction layer,
independent of any dynamical system.

Pipeline role:
    known measure → moments → CD reconstruction → validation
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from experiments.cd_kernel.measures.moments import (
    moments_from_atomic_measure,
    moments_from_density,
)

Array = np.ndarray


@dataclass
class AtomicMeasure:
    angles: Array
    weights: Array

    def moments(self, order: int, normalize: bool = True) -> Array:
        return moments_from_atomic_measure(
            self.angles, self.weights, order=order, normalize=normalize
        )


@dataclass
class AbsolutelyContinuousMeasure:
    density_fn: Callable[[Array], Array]

    def moments(
        self,
        order: int,
        grid_size: int = 4096,
        normalize: bool = True,
    ) -> Array:
        return moments_from_density(
            self.density_fn,
            order=order,
            grid_size=grid_size,
            normalize=normalize,
        )


def uniform_density(angles: Array) -> Array:
    return np.ones_like(angles, dtype=float) / (2.0 * np.pi)


def cosine_density(alpha: float = 0.4):
    if abs(alpha) >= 1.0:
        raise ValueError("use |alpha| < 1 to keep density nonnegative")

    def rho(angles: Array) -> Array:
        return (1.0 + alpha * np.cos(angles)) / (2.0 * np.pi)

    return rho


def wrapped_gaussian_density(center: float, sigma: float):
    def rho(angles: Array) -> Array:
        # crude wrapped Gaussian by summing nearby copies
        total = np.zeros_like(angles, dtype=float)
        for m in range(-2, 3):
            diff = angles - center + 2.0 * np.pi * m
            total += np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
        # normalize numerically later if needed
        return total

    return rho