"""
Baseline CD-kernel variant.

This file defines the baseline algorithm variant as a thin wrapper around
the shared CD-kernel core. It exists so that numerical variants can be
compared cleanly without duplicating the common reconstruction machinery.

Role of this file:
- accept moments or trajectories
- call the shared core reconstruction code
- expose the baseline variant API

This is the canonical “v001” implementation against which later variants
(e.g. tapered, regularized, adaptive) can be compared.
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from methods.cd_kernel.core.kernel import evaluate_cd_kernel_from_moments
from methods.cd_kernel.dynamics.spectral_measure import (
    spectral_measure_data_from_trajectory,
)

Array = np.ndarray
Observable = Callable[[Array], Array]


def fit_cd_kernel_baseline_from_moments(
    moments: Array,
    order: int | None = None,
    grid_size: int = 2048,
    regularization: float = 1e-8,
    normalize_density: bool = True,
):
    """
    Baseline CD-kernel reconstruction from a supplied moment sequence.
    """
    return evaluate_cd_kernel_from_moments(
        moments=moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=normalize_density,
    )


def fit_cd_kernel_baseline(
    X: Array,
    order: int,
    observable: Optional[Observable] = None,
    grid_size: int = 2048,
    regularization: float = 1e-8,
    center: bool = True,
    normalize_moments: bool = True,
    taper: Optional[Array] = None,
    normalize_density: bool = True,
):
    """
    Baseline CD-kernel reconstruction from a trajectory.

    Pipeline:
        trajectory → observable signal → empirical moments → core CD reconstruction
    """
    spec = spectral_measure_data_from_trajectory(
        X,
        order=order,
        observable=observable,
        center=center,
        normalize=normalize_moments,
        taper=taper,
    )

    result = evaluate_cd_kernel_from_moments(
        moments=spec.moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=normalize_density,
    )

    result.metadata["variant"] = "cd_kernel_v001_baseline"
    result.metadata["signal_length"] = len(spec.signal)
    return result