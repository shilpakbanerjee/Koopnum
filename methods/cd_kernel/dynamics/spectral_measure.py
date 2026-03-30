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

This file deliberately avoids embedding system-specific logic; it provides
the reusable pipeline from trajectories to moments, and optional helpers
for running the full CD-kernel reconstruction in one call.

Pipeline role:
    dynamics + observable → signal → moments → CD reconstruction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any
import numpy as np

from methods.common.observables import evaluate_observable
from methods.common.moments import (
    estimate_moments_from_signal,
    hann_taper,
    hamming_taper,
)
from methods.cd_kernel.core.kernel import evaluate_cd_kernel_from_moments

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


def get_taper(
    length: int,
    taper: Optional[str | Array] = None,
) -> Optional[Array]:
    """
    Return a taper/window array, or None.

    Parameters
    ----------
    length:
        Length of the signal.
    taper:
        One of:
        - None
        - "hann"
        - "hamming"
        - an explicit 1D numpy array of matching length
    """
    if taper is None:
        return None

    if isinstance(taper, str):
        taper_lower = taper.lower()
        if taper_lower == "hann":
            return hann_taper(length)
        if taper_lower == "hamming":
            return hamming_taper(length)
        raise ValueError(
            f"Unknown taper '{taper}'. Use None, 'hann', 'hamming', or an array."
        )

    taper_arr = np.asarray(taper, dtype=float)
    if taper_arr.ndim != 1 or taper_arr.shape[0] != length:
        raise ValueError(
            "Explicit taper must be a 1D array with length equal to the signal length"
        )
    return taper_arr


def moments_from_trajectory(
    X: Array,
    order: int,
    observable: Optional[Observable] = None,
    center: bool = True,
    normalize: bool = True,
    taper: Optional[str | Array] = None,
) -> Array:
    """
    Compute an empirical moment sequence from a trajectory by first evaluating
    an observable and then using the measure-side signal-to-moments routine.
    """
    signal = signal_from_trajectory(X, observable=observable)
    taper_arr = get_taper(len(signal), taper=taper)

    return estimate_moments_from_signal(
        signal=signal,
        order=order,
        center=center,
        normalize=normalize,
        taper=taper_arr,
    )


def spectral_measure_data_from_trajectory(
    X: Array,
    order: int,
    observable: Optional[Observable] = None,
    center: bool = True,
    normalize: bool = True,
    taper: Optional[str | Array] = None,
) -> SpectralMeasureData:
    """
    Convenience wrapper that returns the full trajectory / signal / moments package.
    """
    signal = signal_from_trajectory(X, observable=observable)
    taper_arr = get_taper(len(signal), taper=taper)

    moments = estimate_moments_from_signal(
        signal=signal,
        order=order,
        center=center,
        normalize=normalize,
        taper=taper_arr,
    )

    return SpectralMeasureData(
        trajectory=np.asarray(X),
        signal=np.asarray(signal, dtype=np.complex128),
        moments=np.asarray(moments, dtype=np.complex128),
        metadata={
            "order": order,
            "center": center,
            "normalize": normalize,
            "taper_used": None if taper is None else (taper if isinstance(taper, str) else "custom"),
            "signal_length": len(signal),
        },
    )


def reconstruct_spectral_measure_from_trajectory(
    X: Array,
    order: int,
    observable: Optional[Observable] = None,
    center: bool = True,
    normalize_moments: bool = True,
    taper: Optional[str | Array] = None,
    grid_size: int = 2048,
    regularization: float = 1e-8,
    normalize_density: bool = True,
):
    """
    Full end-to-end pipeline:

        trajectory
          → observable signal
          → empirical moments
          → CD-kernel reconstruction

    Returns
    -------
    spec_data:
        SpectralMeasureData object containing trajectory, signal, and moments.
    cd_result:
        CDKernelResult object from the core reconstruction layer.
    """
    spec_data = spectral_measure_data_from_trajectory(
        X=X,
        order=order,
        observable=observable,
        center=center,
        normalize=normalize_moments,
        taper=taper,
    )

    cd_result = evaluate_cd_kernel_from_moments(
        moments=spec_data.moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=normalize_density,
    )

    cd_result.metadata["source"] = "trajectory"
    cd_result.metadata["signal_length"] = spec_data.metadata["signal_length"]
    cd_result.metadata["observable_name"] = (
        getattr(observable, "__name__", str(observable))
        if observable is not None
        else "default_observable"
    )

    return spec_data, cd_result


def reconstruct_spectral_measure_from_system(
    system_fn: Callable[..., Array],
    system_kwargs: Optional[dict[str, Any]] = None,
    order: int = 80,
    observable: Optional[Observable] = None,
    center: bool = True,
    normalize_moments: bool = True,
    taper: Optional[str | Array] = None,
    grid_size: int = 2048,
    regularization: float = 1e-8,
    normalize_density: bool = True,
):
    """
    Convenience wrapper for:

        system generator → trajectory → spectral measure reconstruction

    Parameters
    ----------
    system_fn:
        Callable returning a trajectory array.
    system_kwargs:
        Keyword arguments passed to system_fn.

    Returns
    -------
    trajectory:
        Generated trajectory.
    spec_data:
        SpectralMeasureData
    cd_result:
        CDKernelResult
    """
    if system_kwargs is None:
        system_kwargs = {}

    X = system_fn(**system_kwargs)

    spec_data, cd_result = reconstruct_spectral_measure_from_trajectory(
        X=X,
        order=order,
        observable=observable,
        center=center,
        normalize_moments=normalize_moments,
        taper=taper,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=normalize_density,
    )

    cd_result.metadata["system_name"] = getattr(system_fn, "__name__", str(system_fn))
    cd_result.metadata["system_kwargs"] = system_kwargs

    return X, spec_data, cd_result