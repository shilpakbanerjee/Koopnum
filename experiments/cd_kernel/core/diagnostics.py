"""
Diagnostics for CD-kernel spectral reconstructions.

This module provides quantitative summaries of reconstructed spectral
measures, so that experiments can be compared numerically rather than
only visually.

Two classes of diagnostics are included:

1. Shape / concentration diagnostics:
   - L1 and L2 distance between reconstructed densities
   - peak height and peak concentration
   - entropy and spectral flatness
   - total variation

2. Weak-convergence diagnostics:
   These are based on the fact that weak convergence of measures on the
   compact unit circle is characterized by convergence of integrals against
   continuous test functions. In practice, we test against:
   - Fourier modes exp(i k theta)
   - real trigonometric test functions cos(k theta), sin(k theta)
   - optional user-supplied continuous test functions

This is aligned with the weak-topology viewpoint emphasized in the
Korda–Putinar–Mezić framework for spectral-measure approximation.
"""

from __future__ import annotations

from typing import Callable, Sequence
import numpy as np

from experiments.cd_kernel.core.normalization import circle_integral

Array = np.ndarray
TestFunction = Callable[[Array], Array]


def _validate_same_grid(result_a, result_b) -> None:
    if len(result_a.angles) != len(result_b.angles):
        raise ValueError("Results do not have the same grid length")
    if not np.allclose(result_a.angles, result_b.angles):
        raise ValueError("Results are not defined on the same angular grid")


def _density_array(result) -> Array:
    return np.asarray(result.density_proxy, dtype=float)


def l1_distance_density(result_a, result_b) -> float:
    _validate_same_grid(result_a, result_b)
    diff = np.abs(_density_array(result_a) - _density_array(result_b))
    return circle_integral(diff, result_a.angles)


def l2_distance_density(result_a, result_b) -> float:
    _validate_same_grid(result_a, result_b)
    diff2 = (_density_array(result_a) - _density_array(result_b)) ** 2
    return float(np.sqrt(circle_integral(diff2, result_a.angles)))


def max_peak_height(result) -> float:
    return float(np.max(_density_array(result)))


def min_density_value(result) -> float:
    return float(np.min(_density_array(result)))


def total_variation_density(result) -> float:
    y = _density_array(result)
    return float(np.sum(np.abs(np.diff(y))))


def density_entropy(result, floor: float = 1e-16) -> float:
    y = np.maximum(_density_array(result), floor)
    y = y / circle_integral(y, result.angles)
    integrand = y * np.log(y)
    return float(-circle_integral(integrand, result.angles))


def spectral_flatness(result, floor: float = 1e-16) -> float:
    y = np.maximum(_density_array(result), floor)
    geom = float(np.exp(np.mean(np.log(y))))
    arith = float(np.mean(y))
    if arith <= 0:
        return 0.0
    return geom / arith


def peak_mass_ratio(result, top_k: int = 5, min_separation: int = 12, width: int = 3) -> float:
    y = _density_array(result)
    n = len(y)
    peaks = result.top_peaks(k=top_k, min_separation=min_separation)
    if not peaks:
        return 0.0

    mask = np.zeros(n, dtype=bool)
    for item in peaks:
        idx = int(item["index"])
        left = max(0, idx - width)
        right = min(n, idx + width + 1)
        mask[left:right] = True

    peak_mass = circle_integral(y[mask], result.angles[mask])
    total_mass = circle_integral(y, result.angles)
    if total_mass <= 0:
        return 0.0
    return float(peak_mass / total_mass)


# ------------------------------------------------------------------
# Weak-convergence diagnostics
# ------------------------------------------------------------------

def integral_against_test_function(result, phi: TestFunction) -> complex:
    """
    Numerically compute ∫ phi(theta) dmu(theta), where dmu is represented
    by the reconstructed density proxy on the angular grid.
    """
    angles = np.asarray(result.angles, dtype=float)
    rho = _density_array(result)
    values = np.asarray(phi(angles), dtype=np.complex128)
    integrand = values * rho
    if hasattr(np, "trapezoid"):
        return complex(np.trapezoid(integrand, angles))
    return complex(np.trapz(integrand, angles))


def fourier_mode_integral(result, k: int) -> complex:
    """
    Compute ∫ exp(i k theta) dmu(theta) from the reconstructed density.
    """
    return integral_against_test_function(
        result,
        lambda theta: np.exp(1j * k * theta),
    )


def fourier_test_discrepancy(result_a, result_b, max_mode: int = 10) -> dict:
    """
    Compare two reconstructed measures against Fourier test functions
    exp(i k theta), |k| <= max_mode.

    Returns:
        {
            "max_abs_discrepancy": ...,
            "mean_abs_discrepancy": ...,
            "modewise_abs_discrepancies": np.ndarray,
            "modes": np.ndarray,
        }
    """
    _validate_same_grid(result_a, result_b)

    modes = np.arange(-max_mode, max_mode + 1, dtype=int)
    diffs = np.zeros(len(modes), dtype=float)

    for j, k in enumerate(modes):
        ia = fourier_mode_integral(result_a, k)
        ib = fourier_mode_integral(result_b, k)
        diffs[j] = abs(ia - ib)

    return {
        "max_abs_discrepancy": float(np.max(diffs)),
        "mean_abs_discrepancy": float(np.mean(diffs)),
        "modewise_abs_discrepancies": diffs,
        "modes": modes,
    }


def trig_test_discrepancy(result_a, result_b, max_mode: int = 10) -> dict:
    """
    Compare two reconstructed measures against the real trigonometric
    test family {1, cos(k theta), sin(k theta)}.
    """
    _validate_same_grid(result_a, result_b)

    cos_diffs = np.zeros(max_mode + 1, dtype=float)
    sin_diffs = np.zeros(max_mode + 1, dtype=float)

    for k in range(max_mode + 1):
        ia = integral_against_test_function(result_a, lambda th, kk=k: np.cos(kk * th))
        ib = integral_against_test_function(result_b, lambda th, kk=k: np.cos(kk * th))
        cos_diffs[k] = abs(ia - ib)

        ia = integral_against_test_function(result_a, lambda th, kk=k: np.sin(kk * th))
        ib = integral_against_test_function(result_b, lambda th, kk=k: np.sin(kk * th))
        sin_diffs[k] = abs(ia - ib)

    return {
        "max_cos_discrepancy": float(np.max(cos_diffs)),
        "max_sin_discrepancy": float(np.max(sin_diffs)),
        "mean_cos_discrepancy": float(np.mean(cos_diffs)),
        "mean_sin_discrepancy": float(np.mean(sin_diffs)),
        "cos_discrepancies": cos_diffs,
        "sin_discrepancies": sin_diffs,
        "modes": np.arange(max_mode + 1, dtype=int),
    }


def test_function_discrepancy(
    result_a,
    result_b,
    test_functions: Sequence[TestFunction],
) -> dict:
    """
    Compare two measures against a user-supplied family of continuous
    test functions.
    """
    _validate_same_grid(result_a, result_b)

    diffs = np.zeros(len(test_functions), dtype=float)
    for j, phi in enumerate(test_functions):
        ia = integral_against_test_function(result_a, phi)
        ib = integral_against_test_function(result_b, phi)
        diffs[j] = abs(ia - ib)

    return {
        "max_abs_discrepancy": float(np.max(diffs)) if len(diffs) else 0.0,
        "mean_abs_discrepancy": float(np.mean(diffs)) if len(diffs) else 0.0,
        "functionwise_abs_discrepancies": diffs,
    }


def weak_convergence_summary(result_a, result_b, max_mode: int = 10) -> dict:
    """
    Compact weak-convergence summary based on Fourier and trigonometric
    continuous test functions.

    This is the most directly relevant diagnostic for the weak-topology
    viewpoint used in the Korda–Putinar–Mezić framework.
    """
    fourier = fourier_test_discrepancy(result_a, result_b, max_mode=max_mode)
    trig = trig_test_discrepancy(result_a, result_b, max_mode=max_mode)

    return {
        "fourier_max_abs_discrepancy": fourier["max_abs_discrepancy"],
        "fourier_mean_abs_discrepancy": fourier["mean_abs_discrepancy"],
        "trig_max_cos_discrepancy": trig["max_cos_discrepancy"],
        "trig_max_sin_discrepancy": trig["max_sin_discrepancy"],
        "trig_mean_cos_discrepancy": trig["mean_cos_discrepancy"],
        "trig_mean_sin_discrepancy": trig["mean_sin_discrepancy"],
    }


def summarize_result(result, top_k: int = 5, min_separation: int = 12) -> dict:
    return {
        "max_peak_height": max_peak_height(result),
        "min_density_value": min_density_value(result),
        "total_variation": total_variation_density(result),
        "entropy": density_entropy(result),
        "spectral_flatness": spectral_flatness(result),
        "peak_mass_ratio": peak_mass_ratio(
            result,
            top_k=top_k,
            min_separation=min_separation,
        ),
        "toeplitz_condition_number": float(
            result.metadata.get("toeplitz_condition_number", np.nan)
        ),
    }


def compare_results(result_a, result_b, top_k: int = 5, min_separation: int = 12) -> dict:
    return {
        "l1_distance": l1_distance_density(result_a, result_b),
        "l2_distance": l2_distance_density(result_a, result_b),
        "max_peak_height_a": max_peak_height(result_a),
        "max_peak_height_b": max_peak_height(result_b),
        "flatness_a": spectral_flatness(result_a),
        "flatness_b": spectral_flatness(result_b),
        "entropy_a": density_entropy(result_a),
        "entropy_b": density_entropy(result_b),
        "peak_mass_ratio_a": peak_mass_ratio(
            result_a,
            top_k=top_k,
            min_separation=min_separation,
        ),
        "peak_mass_ratio_b": peak_mass_ratio(
            result_b,
            top_k=top_k,
            min_separation=min_separation,
        ),
    }