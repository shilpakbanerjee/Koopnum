from __future__ import annotations

"""Core experiment logic for unified rotation runners.

Most functions here are intentionally thin wrappers around existing Koopnum
infrastructure so that the refactor can preserve current mathematical behavior
while removing duplication across runners.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from methods.common.systems import generate_planar_rotation, generate_torus_translation
from methods.common.observables import complex_coordinate
from methods.cd_kernel.dynamics.spectral_measure import spectral_measure_data_from_trajectory
from methods.cd_kernel.api import (
    evaluate_cd_kernel_from_moments,
    atomic_mass_proxy_from_kernel,
    koopman_matrix_from_moments,
    companion_koopman_from_moments,
    spectral_summary,
)

from ._rotation_config import RotationRunConfig, ResolvedRotationConfig, resolve_rotation_config


@dataclass
class ObservableSpec:
    func: Callable[[np.ndarray], np.ndarray]
    description: str
    expectation: str
    slug: str
    expected_angles: list[float]
    harmonics: np.ndarray
    coefficients: np.ndarray
    show_expected_single_angle: bool = False


@dataclass
class RotationRunResult:
    config: RotationRunConfig
    resolved: ResolvedRotationConfig
    observable: ObservableSpec
    trajectory: np.ndarray
    signal: np.ndarray
    moments: np.ndarray
    moments_for_cd: np.ndarray
    cd_result: Any
    atomic_proxy: np.ndarray
    detected_peaks: list[dict[str, float | complex]]
    koopman_main: Optional[Any] = None
    koopman_companion: Optional[Any] = None
    koopman_main_summary: Optional[list[dict[str, Any]]] = None
    koopman_companion_summary: Optional[list[dict[str, Any]]] = None
    output_files: dict[str, str] | None = None


# ------------------------------------------------------------------
# observable builders
# ------------------------------------------------------------------

def _normalize_coefficients(coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
    if norm <= 1e-14:
        raise ValueError("Observable coefficients must not all be zero.")
    return coeffs / norm


def _circle_observable_from_harmonics(
    harmonics: np.ndarray,
    coeffs: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    harmonics = np.asarray(harmonics, dtype=int)
    coeffs = _normalize_coefficients(coeffs)

    def observable(X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim != 2:
            raise ValueError(f"Expected trajectory array with ndim 1 or 2, got shape {x.shape}.")

        # Circle rotation uses the first coordinate.
        x0 = x[:, 0]

        values = np.zeros(x0.shape, dtype=np.complex128)
        for c, h in zip(coeffs, harmonics):
            values += c * np.exp(2j * np.pi * h * x0)
        return values

    return observable


def build_observable(cfg: RotationRunConfig, resolved: ResolvedRotationConfig) -> ObservableSpec:
    mode = cfg.observable_mode.lower().strip()

    if resolved.case_name == "planar":
        if mode == "eigenfunction":
            return ObservableSpec(
                func=complex_coordinate(0, 1),
                description="complex_coordinate(0, 1)",
                expectation="single atomic peak near the planar rotation eigenangle",
                slug="eigenfunction",
                expected_angles=[float(resolved.theta_effective or 0.0)],
                harmonics=np.array([1], dtype=int),
                coefficients=_normalize_coefficients(np.array([1.0], dtype=np.complex128)),
                show_expected_single_angle=True,
            )

        if mode == "rich":
            # This is intentionally kept simple for the first integration pass.
            # It should be replaced later if you want a more structured planar-rich observable.
            return ObservableSpec(
                func=lambda X: np.exp(2j * np.pi * np.asarray(X, dtype=float)[:, 0]),
                description="exp(2π i x-coordinate)",
                expectation="multiple harmonic atoms; richer finite Koopman matrix",
                slug="rich",
                expected_angles=[],
                harmonics=np.array([1], dtype=int),
                coefficients=_normalize_coefficients(np.array([1.0], dtype=np.complex128)),
                show_expected_single_angle=False,
            )

        raise ValueError("observable_mode must be 'eigenfunction' or 'rich'")

    # circle cases
    if mode == "eigenfunction":
        harmonics = np.array([cfg.observable_eigenvalue_index], dtype=int)
        coeffs = _normalize_coefficients(np.array([1.0], dtype=np.complex128))
        alpha = float(resolved.alpha_effective or 0.0)
        expected_angles = [float((2.0 * np.pi * harmonics[0] * alpha) % (2.0 * np.pi))]
        return ObservableSpec(
            func=_circle_observable_from_harmonics(harmonics, coeffs),
            description=rf"f(x)=e^{{2\pi i\,{harmonics[0]}x}}",
            expectation="single atom at the selected circle-rotation eigenangle",
            slug="eigenfunction",
            expected_angles=expected_angles,
            harmonics=harmonics,
            coefficients=coeffs,
            show_expected_single_angle=True,
        )

    if mode == "rich":
        harmonics = np.array([1, 2, 3], dtype=int)
        coeffs = _normalize_coefficients(np.array([1.0, 0.35, 0.15], dtype=np.complex128))
        alpha = float(resolved.alpha_effective or 0.0)
        expected_angles = [float((2.0 * np.pi * h * alpha) % (2.0 * np.pi)) for h in harmonics]
        expectation = (
            "finite multi-atomic spectrum for the chosen finite Fourier observable; "
            "for irrational rotation the true atoms may still be numerically smeared under truncation"
        )
        return ObservableSpec(
            func=_circle_observable_from_harmonics(harmonics, coeffs),
            description=r"f(x)=e^{2\pi i x}+0.35\,e^{4\pi i x}+0.15\,e^{6\pi i x}",
            expectation=expectation,
            slug="rich",
            expected_angles=expected_angles,
            harmonics=harmonics,
            coefficients=coeffs,
            show_expected_single_angle=False,
        )

    raise ValueError("observable_mode must be 'eigenfunction' or 'rich'")


# ------------------------------------------------------------------
# trajectory, moments, CD-kernel, Koopman
# ------------------------------------------------------------------

def build_rotation_trajectory(cfg: RotationRunConfig, resolved: ResolvedRotationConfig) -> np.ndarray:
    if resolved.case_name == "planar":
        trajectory = generate_planar_rotation(
            n=cfg.n_traj,
            theta=float(resolved.theta_effective or 0.0),
        )
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] < 2:
            raise RuntimeError(
                f"Planar trajectory must be a 2D array with at least 2 coordinates; got shape {trajectory.shape}."
            )
        return trajectory

    alpha = float(resolved.alpha_effective or 0.0)
    trajectory = generate_torus_translation(
        n=cfg.n_traj,
        omega=np.array([alpha], dtype=float),
        x0=np.array([cfg.x0], dtype=float),
    )
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    return trajectory


def build_moment_data(cfg: RotationRunConfig, trajectory: np.ndarray, observable: ObservableSpec):
    return spectral_measure_data_from_trajectory(
        X=trajectory,
        order=cfg.moment_order,
        observable=observable.func,
        center=False,
        normalize=True,
        taper=None,
    )


def prepare_moments_for_cd(moments: np.ndarray, resolved: ResolvedRotationConfig) -> np.ndarray:
    """Apply convention fixes before CD-kernel evaluation.

    We standardize to the convention expected by the CD-kernel layer:
    if empirical moments are produced with the opposite inner-product ordering,
    conjugate them here so that atomic peaks appear at the mathematically
    expected eigenangles.
    """
    moments_for_cd = np.array(moments, copy=True)

    # Current empirical moment convention appears reversed relative to the
    # expected-angle convention, so conjugate before CD-kernel evaluation.
    moments_for_cd = np.conjugate(moments_for_cd)

    if abs(moments_for_cd[0]) > 1e-14:
        moments_for_cd = moments_for_cd / moments_for_cd[0]

    return moments_for_cd


def evaluate_cd_kernel(cfg: RotationRunConfig, moments_for_cd: np.ndarray):
    cd_result = evaluate_cd_kernel_from_moments(
        moments=moments_for_cd,
        order=cfg.moment_order,
        grid_size=cfg.grid_size,
        regularization=cfg.regularization,
        normalize_density=True,
    )

    if not hasattr(cd_result, "kernel_diag"):
        raise RuntimeError("CD kernel result missing kernel_diag")
    if not hasattr(cd_result, "angles"):
        raise RuntimeError("CD kernel result missing angles")
    if not hasattr(cd_result, "metadata"):
        raise RuntimeError("CD kernel result missing metadata")

    order_used = int(cd_result.metadata["order_used"])
    atomic_proxy = atomic_mass_proxy_from_kernel(cd_result.kernel_diag, order=order_used)
    return cd_result, atomic_proxy


def pick_top_peaks(
    angles: np.ndarray,
    values: np.ndarray,
    k: int = 10,
    min_separation: int = 12,
) -> list[dict[str, float | complex]]:
    y = np.asarray(values, dtype=float)
    candidates: list[tuple[int, float]] = []

    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            candidates.append((i, float(y[i])))

    candidates.sort(key=lambda item: item[1], reverse=True)

    selected: list[dict[str, float | complex]] = []
    taken = np.zeros(len(y), dtype=bool)

    for idx, val in candidates:
        left = max(0, idx - min_separation)
        right = min(len(y), idx + min_separation + 1)
        if np.any(taken[left:right]):
            continue
        taken[left:right] = True
        selected.append(
            {
                "index": idx,
                "angle": float(angles[idx]),
                "value": float(val),
                "point": complex(np.exp(1j * float(angles[idx]))),
            }
        )
        if len(selected) >= k:
            break

    return selected


def compute_koopman_side(cfg: RotationRunConfig, moments: np.ndarray):
    if cfg.koopman_mode == "none":
        return None, None, None, None

    koopman_main = koopman_matrix_from_moments(
        moments=moments,
        order=cfg.koopman_order,
        regularization=cfg.regularization,
        solve_method="solve",
    )
    koopman_companion = companion_koopman_from_moments(
        moments=moments,
        order=cfg.koopman_order,
    )
    main_summary = spectral_summary(koopman_main, top_k=10)
    companion_summary = spectral_summary(koopman_companion, top_k=10)
    return koopman_main, koopman_companion, main_summary, companion_summary


def run_rotation_experiment(cfg: RotationRunConfig) -> RotationRunResult:
    resolved = resolve_rotation_config(cfg)
    observable = build_observable(cfg, resolved)
    trajectory = build_rotation_trajectory(cfg, resolved)

    spec_data = build_moment_data(cfg, trajectory, observable)
    moments_for_cd = prepare_moments_for_cd(spec_data.moments, resolved)

    cd_result, atomic_proxy = evaluate_cd_kernel(cfg, moments_for_cd)
    peaks = pick_top_peaks(cd_result.angles, atomic_proxy)

    koopman_main, koopman_companion, main_summary, companion_summary = compute_koopman_side(
        cfg, spec_data.moments
    )

    try:
        top_peak_value = max(p["value"] for p in peaks) if peaks else "NA"
    except Exception:
        top_peak_value = "NA"


    print("\n=== Rotation experiment summary ===")
    print("case:", resolved.case_name)
    if getattr(resolved, "alpha_effective", None) is not None:
        print("alpha:", resolved.alpha_effective)
    if getattr(resolved, "theta_effective", None) is not None:
        print("theta:", resolved.theta_effective)
    print("observable:", observable.description)
    print("expectation:", observable.expectation)
    print("toeplitz condition number:", cd_result.metadata.get("toeplitz_condition_number", "NA"))
    print("top peak value:", top_peak_value)

    return RotationRunResult(
        config=cfg,
        resolved=resolved,
        observable=observable,
        trajectory=trajectory,
        signal=spec_data.signal,
        moments=spec_data.moments,
        moments_for_cd=moments_for_cd,
        cd_result=cd_result,
        atomic_proxy=atomic_proxy,
        detected_peaks=peaks,
        koopman_main=koopman_main,
        koopman_companion=koopman_companion,
        koopman_main_summary=main_summary,
        koopman_companion_summary=companion_summary,
        output_files={},
    )