from __future__ import annotations

"""Weak-convergence utilities for atomic spectral measures on the unit circle.

This module is intentionally lightweight and exact for the finite-atomic
rotation experiments currently used in the unified rotation runner.

Main use:
- build exact atomic spectral measures for finite Fourier observables
- compare irrational rotation against rational approximants
- evaluate weak convergence through moments and continuous test functions
"""

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


_TWO_PI = 2.0 * np.pi


@dataclass
class AtomicMeasureOnCircle:
    angles: np.ndarray   # shape (m,), values in [0, 2pi)
    weights: np.ndarray  # shape (m,), sum to 1


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("Weights must have positive total mass.")
    return w / total


def circular_distance_array(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(np.asarray(a, dtype=float) - float(b)) % _TWO_PI
    return np.minimum(d, _TWO_PI - d)


def merge_close_atoms(
    angles: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-12,
) -> AtomicMeasureOnCircle:
    """Merge atoms whose angles coincide up to tolerance.

    This is especially important for rational rotations, where different
    harmonics may collapse to the same eigenangle modulo 2pi.
    """
    ang = np.mod(np.asarray(angles, dtype=float), _TWO_PI)
    w = normalize_weights(weights)

    order = np.argsort(ang)
    ang = ang[order]
    w = w[order]

    merged_angles: list[float] = []
    merged_weights: list[float] = []

    for a, mass in zip(ang, w):
        if not merged_angles:
            merged_angles.append(float(a))
            merged_weights.append(float(mass))
            continue

        if abs(float(a) - merged_angles[-1]) < tol:
            merged_weights[-1] += float(mass)
        else:
            merged_angles.append(float(a))
            merged_weights.append(float(mass))

    # Also merge wrap-around if first and last are effectively the same point.
    if len(merged_angles) >= 2:
        wrap_gap = min(
            abs(merged_angles[0] - merged_angles[-1]),
            _TWO_PI - abs(merged_angles[0] - merged_angles[-1]),
        )
        if wrap_gap < tol:
            merged_weights[0] += merged_weights[-1]
            merged_angles.pop(-1)
            merged_weights.pop(-1)

    merged = AtomicMeasureOnCircle(
        angles=np.mod(np.array(merged_angles, dtype=float), _TWO_PI),
        weights=normalize_weights(np.array(merged_weights, dtype=float)),
    )
    return merged


def build_rotation_measure(
    alpha: float,
    harmonics: np.ndarray,
    coefficients: np.ndarray,
    tol: float = 1e-12,
) -> AtomicMeasureOnCircle:
    """Build the exact spectral measure for a finite Fourier observable.

    Observable:
        f(x) = sum_j c_j exp(2 pi i h_j x)

    Measure:
        mu = sum_j |c_j|^2 delta_{exp(2 pi i h_j alpha)}
    with masses merged when angles coincide modulo 2pi.
    """
    h = np.asarray(harmonics, dtype=int)
    c = np.asarray(coefficients, dtype=np.complex128)

    if h.ndim != 1 or c.ndim != 1 or len(h) != len(c):
        raise ValueError("harmonics and coefficients must be 1D arrays of the same length.")

    weights = normalize_weights(np.abs(c) ** 2)
    angles = np.mod(_TWO_PI * h * float(alpha), _TWO_PI)

    return merge_close_atoms(angles, weights, tol=tol)


def integrate_test_function(
    measure: AtomicMeasureOnCircle,
    phi: Callable[[np.ndarray], np.ndarray],
) -> float:
    vals = np.asarray(phi(measure.angles), dtype=float)
    return float(np.sum(measure.weights * vals))


def complex_moment(measure: AtomicMeasureOnCircle, k: int) -> complex:
    return complex(np.sum(measure.weights * np.exp(1j * int(k) * measure.angles)))


def real_trig_moment(measure: AtomicMeasureOnCircle, k: int) -> tuple[float, float]:
    theta = np.asarray(measure.angles, dtype=float)
    cos_part = float(np.sum(measure.weights * np.cos(int(k) * theta)))
    sin_part = float(np.sum(measure.weights * np.sin(int(k) * theta)))
    return cos_part, sin_part


def gaussian_bump(theta0: float, sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    theta0 = float(theta0)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    def phi(theta: np.ndarray) -> np.ndarray:
        d = circular_distance_array(theta, theta0)
        return np.exp(-(d ** 2) / (2.0 * sigma ** 2))

    return phi


def build_default_test_functions(
    target_measure: AtomicMeasureOnCircle,
    bump_sigma: float = 0.25,
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    tests: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "cos_1": lambda th: np.cos(th),
        "sin_1": lambda th: np.sin(th),
        "cos_2": lambda th: np.cos(2.0 * th),
        "sin_2": lambda th: np.sin(2.0 * th),
        "cos_3": lambda th: np.cos(3.0 * th),
        "sin_3": lambda th: np.sin(3.0 * th),
        "bump_pi": gaussian_bump(np.pi, bump_sigma),
    }

    for idx, ang in enumerate(target_measure.angles, start=1):
        tests[f"bump_atom_{idx}"] = gaussian_bump(float(ang), bump_sigma)

    return tests


def compare_rotation_measures_weakly(
    target_alpha: float,
    approximants: Iterable[tuple[int, int]],
    harmonics: np.ndarray,
    coefficients: np.ndarray,
    max_moment_order: int = 5,
    bump_sigma: float = 0.25,
) -> dict:
    """Compare rational approximants to the target irrational rotation weakly.

    Returns a dictionary with:
    - target_measure
    - target_moments
    - target_tests
    - rows: one per approximant with moment/test-function errors
    """
    target_measure = build_rotation_measure(
        alpha=float(target_alpha),
        harmonics=harmonics,
        coefficients=coefficients,
    )

    tests = build_default_test_functions(target_measure, bump_sigma=bump_sigma)

    target_moments = {
        k: complex_moment(target_measure, k)
        for k in range(1, int(max_moment_order) + 1)
    }
    target_tests = {
        name: integrate_test_function(target_measure, phi)
        for name, phi in tests.items()
    }

    rows: list[dict] = []

    original_harmonic_count = int(len(np.asarray(harmonics)))

    for p, q in approximants:
        alpha_n = float(p) / float(q)
        mu_n = build_rotation_measure(
            alpha=alpha_n,
            harmonics=harmonics,
            coefficients=coefficients,
        )

        row: dict[str, float | int] = {
            "p": int(p),
            "q": int(q),
            "alpha_n": alpha_n,
            "alpha_error": abs(alpha_n - float(target_alpha)),
            "num_atoms": int(len(mu_n.angles)),
        }

        row.update(collapse_metrics(mu_n, num_original_harmonics=original_harmonic_count))

        # Raw atom info for quick inspection
        for j, (ang, wt) in enumerate(zip(mu_n.angles, mu_n.weights), start=1):
            row[f"atom_{j}_angle"] = float(ang)
            row[f"atom_{j}_weight"] = float(wt)

        # Moment errors
        for k in range(1, int(max_moment_order) + 1):
            mk_n = complex_moment(mu_n, k)
            row[f"moment_{k}_abs_error"] = abs(mk_n - target_moments[k])

        # Continuous-test-function errors
        for name, phi in tests.items():
            val_n = integrate_test_function(mu_n, phi)
            row[f"test_{name}_abs_error"] = abs(val_n - target_tests[name])

        rows.append(row)

    return {
        "target_measure": target_measure,
        "target_moments": target_moments,
        "target_tests": target_tests,
        "rows": rows,
    }


def smoothed_density_on_grid(
    measure: AtomicMeasureOnCircle,
    angle_grid: np.ndarray,
    sigma: float = 0.20,
) -> np.ndarray:
    """Return a periodic Gaussian-smoothed density-like profile on the grid."""
    grid = np.asarray(angle_grid, dtype=float)
    y = np.zeros_like(grid, dtype=float)

    for ang, wt in zip(measure.angles, measure.weights):
        d = circular_distance_array(grid, float(ang))
        y += float(wt) * np.exp(-(d ** 2) / (2.0 * sigma ** 2))

    return y


def spectral_entropy(weights: np.ndarray, eps: float = 1e-15) -> float:
    """Return Shannon entropy of normalized atom weights.

    If w = (w_1,...,w_r) with sum w_i = 1, then
        H(w) = -sum_i w_i log w_i.

    Interpretation:
    - H = 0 for complete concentration on one atom
    - H = log(r) for r equal-mass atoms
    """
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    w = w / total
    w_safe = np.clip(w, eps, None)
    return float(-np.sum(w * np.log(w_safe)))


def effective_atom_count(weights: np.ndarray, eps: float = 1e-15) -> float:
    """Return exp(entropy), interpreted as effective number of active atoms."""
    return float(np.exp(spectral_entropy(weights, eps=eps)))


def sorted_weights_desc(weights: np.ndarray) -> np.ndarray:
    """Return normalized weights in descending order."""
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    w = w / total
    return np.sort(w)[::-1]


def top_mass_fraction(weights: np.ndarray, top_k: int = 1) -> float:
    """Mass carried by the largest top_k atoms."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    w = sorted_weights_desc(weights)
    return float(np.sum(w[:top_k]))


def concentration_ratio(weights: np.ndarray) -> float:
    """Largest weight divided by second-largest weight.

    Returns +inf if there is only one atom or the second-largest weight is 0.
    """
    w = sorted_weights_desc(weights)
    if len(w) == 0:
        raise ValueError("weights must not be empty")
    if len(w) == 1:
        return float("inf")
    if w[1] <= 1e-15:
        return float("inf")
    return float(w[0] / w[1])


def collision_count(num_original_harmonics: int, num_atoms_after_merge: int) -> int:
    """How many original harmonic contributions were merged away."""
    return int(num_original_harmonics) - int(num_atoms_after_merge)


def collapse_metrics(measure, num_original_harmonics: int) -> dict:
    """Return interpretable metrics for spectral collapse.

    Parameters
    ----------
    measure:
        AtomicMeasureOnCircle with fields `angles` and `weights`.
    num_original_harmonics:
        Number of harmonic terms in the original observable before merging.

    Returns
    -------
    dict
        Contains:
        - num_atoms
        - collision_count
        - spectral_entropy
        - effective_atom_count
        - top_1_mass
        - top_2_mass
        - concentration_ratio
    """
    w = np.asarray(measure.weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("measure must have positive total mass")
    w = w / total

    num_atoms = int(len(w))

    return {
        "num_atoms": num_atoms,
        "collision_count": collision_count(num_original_harmonics, num_atoms),
        "spectral_entropy": spectral_entropy(w),
        "effective_atom_count": effective_atom_count(w),
        "top_1_mass": top_mass_fraction(w, top_k=1),
        "top_2_mass": top_mass_fraction(w, top_k=min(2, num_atoms)),
        "concentration_ratio": concentration_ratio(w),
    }