from __future__ import annotations
import numpy as np

Array = np.ndarray


def cantor_points(stage: int) -> Array:
    """
    Generate Cantor points in [0,1] at given stage.
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


def cantor_measure_on_circle(stage: int):
    """
    Return atomic approximation of Cantor measure on unit circle.

    Returns:
        angles, weights
    """
    x = cantor_points(stage)
    angles = 2.0 * np.pi * x
    weights = np.ones_like(angles) / len(angles)
    return angles, weights


def moments_from_atomic(angles: Array, weights: Array, order: int) -> Array:
    """
    Compute moments m_k = ∫ e^{ikθ} dμ from atomic measure.
    """
    moments = np.zeros(order + 1, dtype=np.complex128)

    for k in range(order + 1):
        moments[k] = np.sum(weights * np.exp(1j * k * angles))

    return moments