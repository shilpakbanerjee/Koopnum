from __future__ import annotations

"""Configuration and parameter-resolution helpers for rotation experiments.

This file is intentionally a skeleton: the public API is stable enough to begin
refactoring, but several project-specific hooks still need to be wired to the
exact Koopnum code paths.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, Optional

import math
import numpy as np

RotationKind = Literal["planar", "circle"]
AlphaMode = Literal["rational", "irrational_float", "irrational_approximant"]
ObservableMode = Literal["eigenfunction", "rich"]
ApproxFamily = Literal["continued_fraction", "fibonacci"]
KoopmanMode = Literal["none", "galerkin", "all"]
CompareMode = Literal[
    "irrational_vs_approximants",
    "observable_modes",
    "rational_family",
]


IRRATIONAL_PRESETS: dict[str, float] = {
    "golden_conjugate": (math.sqrt(5.0) - 1.0) / 2.0,
    "sqrt2_minus_1": math.sqrt(2.0) - 1.0,
    "sqrt3_minus_1": math.sqrt(3.0) - 1.0,
}


@dataclass
class RotationRunConfig:
    # high-level system selection
    rotation_kind: RotationKind = "circle"
    alpha_mode: AlphaMode = "irrational_float"

    # direct angle data
    alpha: Optional[float] = None
    theta: Optional[float] = None  # planar angle in radians
    p: Optional[int] = None
    q: Optional[int] = None

    # irrational approximation target
    target_alpha: Optional[float] = None
    target_alpha_name: Optional[str] = "golden_conjugate"
    approx_family: ApproxFamily = "fibonacci"
    approx_index: Optional[int] = None

    # observable selection
    observable_mode: ObservableMode = "rich"
    observable_eigenvalue_index: int = 1

    # numerics
    x0: float = 0.123456789
    n_traj: int = 4000
    moment_order: int = 120
    grid_size: int = 4096
    regularization: float = 1e-8

    # Koopman-side options
    koopman_mode: KoopmanMode = "none"
    koopman_order: int = 30

    # outputs
    output_dir: str = "outputs/dynamics/rotation"
    run_name: Optional[str] = None
    make_plots: bool = True
    save_npz: bool = True


@dataclass
class ResolvedRotationConfig:
    case_name: str
    alpha_effective: Optional[float]
    alpha_label: str
    theta_effective: Optional[float]
    rational_data: Optional[tuple[int, int]]
    is_exact_periodic: bool
    expected_period: Optional[int]


@dataclass
class RotationComparisonConfig:
    compare_mode: CompareMode = "irrational_vs_approximants"
    base_config: RotationRunConfig = field(default_factory=RotationRunConfig)
    approx_indices: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    observable_modes: list[ObservableMode] = field(default_factory=lambda: ["eigenfunction", "rich"])
    rational_pairs: list[tuple[int, int]] = field(default_factory=list)
    output_dir: str = "outputs/dynamics/rotation_comparison"
    make_plots: bool = True
    save_csv: bool = True


def infer_rotation_case(cfg: RotationRunConfig) -> str:
    if cfg.rotation_kind == "planar":
        return "planar"
    if cfg.rotation_kind != "circle":
        raise ValueError(f"Unsupported rotation_kind={cfg.rotation_kind!r}")
    if cfg.alpha_mode == "rational":
        return "circle_rational"
    if cfg.alpha_mode == "irrational_float":
        return "circle_irrational_float"
    if cfg.alpha_mode == "irrational_approximant":
        return "circle_single_approximant"
    raise ValueError(f"Unsupported alpha_mode={cfg.alpha_mode!r}")


def get_named_irrational(name: str) -> float:
    try:
        return IRRATIONAL_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown irrational preset {name!r}") from exc


def _resolve_target_alpha(cfg: RotationRunConfig) -> tuple[float, str]:
    if cfg.target_alpha is not None:
        return float(cfg.target_alpha), cfg.target_alpha_name or "target_alpha"
    if cfg.target_alpha_name is not None:
        return get_named_irrational(cfg.target_alpha_name), cfg.target_alpha_name
    if cfg.alpha is not None:
        return float(cfg.alpha), "alpha"
    raise ValueError("No irrational target was provided.")


def fibonacci_convergents(count: int) -> list[tuple[int, int]]:
    if count < 1:
        raise ValueError("count must be positive")
    fib = [1, 1]
    while len(fib) < count + 2:
        fib.append(fib[-1] + fib[-2])
    return [(fib[n - 2], fib[n - 1]) for n in range(2, count + 2)]


def continued_fraction_convergent(alpha: float, index: int) -> tuple[int, int]:
    """Return the index-th convergent using Python's Fraction helper.

    This is intentionally lightweight for the skeleton. If you later want exact
    control over all intermediate convergents, replace this with a direct
    continued-fraction implementation.
    """
    if index < 1:
        raise ValueError("approx_index must be positive")
    frac = Fraction(alpha).limit_denominator(10 ** (index + 1))
    return frac.numerator, frac.denominator


def rational_approximant(alpha: float, family: ApproxFamily, index: int) -> tuple[int, int]:
    if family == "fibonacci":
        pairs = fibonacci_convergents(index)
        return pairs[-1]
    if family == "continued_fraction":
        return continued_fraction_convergent(alpha, index)
    raise ValueError(f"Unsupported approx_family={family!r}")


def resolve_rotation_config(cfg: RotationRunConfig) -> ResolvedRotationConfig:
    case_name = infer_rotation_case(cfg)

    if case_name == "planar":
        theta = float(cfg.theta if cfg.theta is not None else 0.35)
        return ResolvedRotationConfig(
            case_name=case_name,
            alpha_effective=None,
            alpha_label=f"theta={theta:.12g}",
            theta_effective=theta,
            rational_data=None,
            is_exact_periodic=False,
            expected_period=None,
        )

    if case_name == "circle_rational":
        if cfg.p is None or cfg.q is None:
            raise ValueError("Rational circle rotation requires p and q.")
        p = int(cfg.p)
        q = int(cfg.q)
        if q <= 0:
            raise ValueError("q must be positive")
        alpha = p / q
        return ResolvedRotationConfig(
            case_name=case_name,
            alpha_effective=alpha,
            alpha_label=f"{p}/{q}",
            theta_effective=None,
            rational_data=(p, q),
            is_exact_periodic=True,
            expected_period=q,
        )

    if case_name == "circle_irrational_float":
        alpha = float(cfg.alpha) if cfg.alpha is not None else get_named_irrational(cfg.target_alpha_name or "golden_conjugate")
        label = cfg.target_alpha_name or "irrational_float"
        return ResolvedRotationConfig(
            case_name=case_name,
            alpha_effective=alpha,
            alpha_label=label,
            theta_effective=None,
            rational_data=None,
            is_exact_periodic=False,
            expected_period=None,
        )

    if case_name == "circle_single_approximant":
        alpha_target, target_name = _resolve_target_alpha(cfg)
        if cfg.approx_index is None:
            raise ValueError("approx_index is required for irrational_approximant mode.")
        p, q = rational_approximant(alpha_target, cfg.approx_family, cfg.approx_index)
        return ResolvedRotationConfig(
            case_name=case_name,
            alpha_effective=p / q,
            alpha_label=f"{target_name}_approx_{cfg.approx_index}_{p}/{q}",
            theta_effective=None,
            rational_data=(p, q),
            is_exact_periodic=True,
            expected_period=q,
        )

    raise RuntimeError("Unreachable case in resolve_rotation_config().")


def trajectory_length_for_denominator(q: int, min_cycles: int = 24, floor: int = 3000) -> int:
    return max(floor, min_cycles * q)


def circular_distance(a: float, b: float) -> float:
    return float(abs(np.angle(np.exp(1j * (a - b)))))
