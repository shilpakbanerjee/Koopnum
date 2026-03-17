"""
Spectral-measure interface for the dynamical side of the CD-kernel pipeline.

This module converts a dynamical trajectory and observable into the
moment sequence of the associated Koopman spectral measure. It acts as
the bridge between the dynamics layer and the measure-reconstruction layer.

Workflow:
    trajectory X
        → observable signal f(X)
        → empirical moments m_k
        → hand off to measure-side CD reconstruction

This file deliberately avoids implementing the reconstruction itself;
that functionality belongs to the measure-side modules.

Pipeline role:
    dynamics + observable → signal → moments
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from experiments.cd_kernel.dynamics.observables import evaluate_observable
from experiments.cd_kernel.measures.moments import estimate_moments_from_signal

Array = np.ndarray
Observable = Callable[[Array], Array]


@dataclass
class SpectralMeasureData:
    trajectory: Array
    signal: Array
    moments: Array
    metadata: dict


def signal_from_trajectory(
    X: Array,
    observable: Optional[Observable] = None,
) -> Array:
    """
    Evaluate the chosen observable on a trajectory and return the
    resulting scalar signal.
    """
    return evaluate_observable(X, observable=observable)


def moments_from_trajectory(
    X: Array,
    order: int,
    observable: Optional[Observable] = None,
    center: bool = True,
    normalize: bool = True,
    taper: Optional[Array] = None,
) -> Array:
    """
    Compute empirical moment sequence from a trajectory by first evaluating
    an observable and then using the measure-side signal-to-moments routine.
    """
    signal = signal_from_trajectory(X, observable=observable)
    return estimate_moments_from_signal(
        signal=signal,
        order=order,
        center=center,
        normalize=normalize,
        taper=taper,
    )


def spectral_measure_data_from_trajectory(
    X: Array,
    order: int,
    observable: Optional[Observable] = None,
    center: bool = True,
    normalize: bool = True,
    taper: Optional[Array] = None,
) -> SpectralMeasureData:
    """
    Convenience wrapper that returns the full trajectory / signal / moments package.
    """
    signal = signal_from_trajectory(X, observable=observable)
    moments = estimate_moments_from_signal(
        signal=signal,
        order=order,
        center=center,
        normalize=normalize,
        taper=taper,
    )
    return SpectralMeasureData(
        trajectory=np.asarray(X),
        signal=signal,
        moments=moments,
        metadata={
            "order": order,
            "center": center,
            "normalize": normalize,
            "taper_used": taper is not None,
            "signal_length": len(signal),
        },
    )