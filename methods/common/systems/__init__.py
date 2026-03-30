"""
Benchmark dynamical systems for CD-kernel spectral experiments.

This module provides simple trajectory generators for deterministic
dynamical systems whose Koopman spectral type is known or expected.
These systems are used to test the full pipeline:
    dynamics → observable signal → moments → CD reconstruction.

The emphasis is on clean benchmark examples rather than exhaustive
physical realism.

Included examples:
- planar rotation (pure point spectrum)
- torus translation (pure point spectrum)
- doubling map on the circle (typically absolutely continuous)
- Arnold cat map on the torus (hyperbolic / continuous spectrum)

Pipeline role:
    dynamical system → trajectory
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def generate_planar_rotation(
    n: int,
    theta: float,
    x0: Array | None = None,
) -> Array:
    """
    Generate a trajectory in R^2 under rotation by angle theta.
    """
    if n < 1:
        raise ValueError("n must be positive")

    if x0 is None:
        x0 = np.array([1.0, 0.0], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (2,):
            raise ValueError("x0 must be shape (2,)")

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ],
        dtype=float,
    )

    X = np.zeros((n, 2), dtype=float)
    X[0] = x0
    for k in range(n - 1):
        X[k + 1] = R @ X[k]
    return X


def generate_torus_translation(
    n: int,
    omega: Array,
    x0: Array | None = None,
) -> Array:
    """
    Generate a trajectory on the d-dimensional torus [0,1)^d:
        x_{k+1} = x_k + omega mod 1.
    """
    omega = np.asarray(omega, dtype=float)
    d = omega.size
    if d < 1:
        raise ValueError("omega must be nonempty")
    if x0 is None:
        x0 = np.zeros(d, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (d,):
            raise ValueError("x0 shape must match omega")

    X = np.zeros((n, d), dtype=float)
    X[0] = np.mod(x0, 1.0)
    for k in range(n - 1):
        X[k + 1] = np.mod(X[k] + omega, 1.0)
    return X


def generate_doubling_map(
    n: int,
    x0: float = 0.123456789,
) -> Array:
    """
    Generate the 1D doubling map on the circle:
        x_{k+1} = 2 x_k mod 1.
    """
    X = np.zeros(n, dtype=float)
    X[0] = float(x0) % 1.0
    for k in range(n - 1):
        X[k + 1] = (2.0 * X[k]) % 1.0
    return X


def generate_cat_map(
    n: int,
    x0: Array | None = None,
    A: Array | None = None,
) -> Array:
    """
    Generate the Arnold cat map (or a user-specified 2x2 integer hyperbolic map)
    on the 2-torus:
        x_{k+1} = A x_k mod 1.

    Default:
        A = [[2, 1],
             [1, 1]]
    """
    if x0 is None:
        x0 = np.array([0.123456789, 0.314159265], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (2,):
            raise ValueError("x0 must be shape (2,)")

    if A is None:
        A = np.array([[2, 1], [1, 1]], dtype=int)
    else:
        A = np.asarray(A, dtype=int)
        if A.shape != (2, 2):
            raise ValueError("A must be shape (2,2)")

    X = np.zeros((n, 2), dtype=float)
    X[0] = np.mod(x0, 1.0)
    for k in range(n - 1):
        X[k + 1] = np.mod(A @ X[k], 1.0)
    return X


def circle_embedding_from_angle_trajectory(theta_traj: Array) -> Array:
    """
    Embed a 1D angle trajectory into R^2 via (cos(2πx), sin(2πx)).
    Useful for visualizing circle maps as planar trajectories.
    """
    theta_traj = np.asarray(theta_traj, dtype=float).reshape(-1)
    return np.column_stack(
        [
            np.cos(2.0 * np.pi * theta_traj),
            np.sin(2.0 * np.pi * theta_traj),
        ]
    )