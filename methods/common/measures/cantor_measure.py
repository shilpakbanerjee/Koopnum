from __future__ import annotations
import numpy as np

Array = np.ndarray


def cantor_points(stage: int) -> Array:
    """
    Generate the standard finite-stage middle-third Cantor points in [0, 1].

    At stage n there are 2^n points, corresponding to base-3 expansions
    using only digits 0 and 2.
    """
    if stage < 0:
        raise ValueError("stage must be >= 0")

    points = np.array([0.0], dtype=float)

    for _ in range(stage):
        points = np.concatenate([
            points / 3.0,
            (points + 2.0) / 3.0,
        ])

    return points


def cantor_measure_on_circle(stage: int) -> tuple[Array, Array]:
    """
    Return a finite-stage atomic approximation of the Cantor measure
    pushed to the unit circle.

    Parameters
    ----------
    stage:
        Cantor construction stage.

    Returns
    -------
    angles, weights
        angles in [0, 2pi), equal atomic weights summing to 1.
    """
    x = cantor_points(stage)
    angles = 2.0 * np.pi * x
    weights = np.ones_like(angles, dtype=float) / len(angles)
    return angles, weights


def moments_from_atomic(angles: Array, weights: Array, order: int) -> Array:
    """
    Compute moments
        m_k = sum_j w_j exp(i k theta_j),   k = 0, ..., order
    for an atomic measure on the unit circle.
    """
    angles = np.asarray(angles, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if angles.shape != weights.shape:
        raise ValueError("angles and weights must have the same shape")
    if order < 0:
        raise ValueError("order must be >= 0")

    moments = np.zeros(order + 1, dtype=np.complex128)
    for k in range(order + 1):
        moments[k] = np.sum(weights * np.exp(1j * k * angles))
    return moments