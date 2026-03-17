"""
Observable definitions for the CD-kernel dynamical pipeline.

This module contains reusable observable maps that turn a trajectory
in state space into a scalar signal suitable for spectral analysis.
Given a trajectory
    X = (x_0, x_1, ..., x_N),
an observable f produces the scalar time series
    f(x_0), f(x_1), ..., f(x_N).

These observables are used to define the Koopman spectral measure
associated with a dynamical system and a chosen scalar function.

The functionality here is independent of the measure-reconstruction
machinery: it only maps state trajectories to scalar signals.

Pipeline role:
    trajectory → observable signal
"""

from __future__ import annotations

from typing import Callable, Sequence
import numpy as np

Array = np.ndarray
Observable = Callable[[Array], Array]


def _as_2d_trajectory(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("Trajectory must be a 1D or 2D array")
    return X


def first_coordinate(X: Array) -> Array:
    X = _as_2d_trajectory(X)
    return X[:, 0].astype(np.complex128)


def second_coordinate(X: Array) -> Array:
    X = _as_2d_trajectory(X)
    if X.shape[1] < 2:
        raise ValueError("Trajectory does not have a second coordinate")
    return X[:, 1].astype(np.complex128)


def coordinate(index: int) -> Observable:
    def obs(X: Array) -> Array:
        X2 = _as_2d_trajectory(X)
        if index < 0 or index >= X2.shape[1]:
            raise ValueError(f"Coordinate index {index} out of range")
        return X2[:, index].astype(np.complex128)

    return obs


def complex_coordinate(i: int = 0, j: int = 1) -> Observable:
    """
    Return x_i + i x_j as a complex observable.
    Useful for planar rotations and torus-like examples.
    """
    def obs(X: Array) -> Array:
        X2 = _as_2d_trajectory(X)
        if i < 0 or i >= X2.shape[1] or j < 0 or j >= X2.shape[1]:
            raise ValueError("Coordinate index out of range")
        return X2[:, i].astype(np.complex128) + 1j * X2[:, j].astype(np.complex128)

    return obs


def linear_observable(weights: Sequence[complex]) -> Observable:
    """
    Return the linear observable f(x) = <w, x>.
    """
    w = np.asarray(weights, dtype=np.complex128)

    def obs(X: Array) -> Array:
        X2 = _as_2d_trajectory(X)
        if X2.shape[1] != w.size:
            raise ValueError("weights length must match trajectory dimension")
        return X2.astype(np.complex128) @ w

    return obs


def quadratic_monomial(i: int, j: int) -> Observable:
    """
    Return the observable f(x) = x_i x_j.
    """
    def obs(X: Array) -> Array:
        X2 = _as_2d_trajectory(X)
        if i < 0 or i >= X2.shape[1] or j < 0 or j >= X2.shape[1]:
            raise ValueError("Coordinate index out of range")
        return (X2[:, i] * X2[:, j]).astype(np.complex128)

    return obs


def cosine_of_coordinate(index: int = 0) -> Observable:
    def obs(X: Array) -> Array:
        X2 = _as_2d_trajectory(X)
        if index < 0 or index >= X2.shape[1]:
            raise ValueError("Coordinate index out of range")
        return np.cos(X2[:, index]).astype(np.complex128)

    return obs


def sine_of_coordinate(index: int = 0) -> Observable:
    def obs(X: Array) -> Array:
        X2 = _as_2d_trajectory(X)
        if index < 0 or index >= X2.shape[1]:
            raise ValueError("Coordinate index out of range")
        return np.sin(X2[:, index]).astype(np.complex128)

    return obs


def evaluate_observable(X: Array, observable: Observable | None = None) -> Array:
    """
    Evaluate an observable on a trajectory.
    If observable is None, use the first coordinate.
    """
    X2 = _as_2d_trajectory(X)
    if observable is None:
        return first_coordinate(X2)
    values = np.asarray(observable(X2), dtype=np.complex128)
    if values.ndim != 1 or values.shape[0] != X2.shape[0]:
        raise ValueError("Observable must return a 1D array of length len(trajectory)")
    return values