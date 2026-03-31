from __future__ import annotations

"""Unified single-run rotation runner.

This file is the main entry point for one rotation experiment. It delegates all
mathematical work to the shared rotation core and keeps only:
- configuration selection
- execution
- basic reporting / saving hooks

The current refactor goal is to make this runner reproduce the old single-run
rotation behavior while keeping the comparison logic separate.
"""

from copy import deepcopy
from pathlib import Path

from ._rotation_config import RotationRunConfig
from ._rotation_core import run_rotation_experiment
from ._rotation_reporting import (
    print_rotation_report,
    compare_peaks_to_expected_angles,
    save_rotation_outputs,
    plot_rotation_result,
)


# ---------------------------------------------------------------------
# Main user-facing config
# ---------------------------------------------------------------------

CONFIG = RotationRunConfig(
    # "circle" or "planar"
    rotation_kind="circle",

    # "rational", "irrational_float", "irrational_approximant"
    alpha_mode="irrational_float",

    # circle parameters
    alpha=(5.0**0.5 - 1.0) / 2.0,
    p=None,
    q=None,

    # planar parameter
    theta=None,

    # irrational approximant target
    target_alpha=None,
    target_alpha_name="golden_conjugate",
    approx_family="continued_fraction",
    approx_index=5,

    # observable
    observable_mode="eigenfunction",   # "eigenfunction" or "rich"
    observable_eigenvalue_index=1,

    # numerics
    x0=0.123456789,
    n_traj=4000,
    moment_order=120,
    grid_size=512,
    regularization=1e-8,

    # Koopman side
    koopman_mode="none",               # "none", "galerkin", "all"
    koopman_order=30,

    # outputs
    output_dir="outputs/rotation",
    run_name=None,
    make_plots=True,
    save_npz=True,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _default_run_name(cfg: RotationRunConfig) -> str:
    if cfg.rotation_kind == "planar":
        obs = cfg.observable_mode
        return f"rotation_planar_{obs}"

    if cfg.alpha_mode == "rational":
        obs = cfg.observable_mode
        return f"rotation_circle_rational_{cfg.p}_{cfg.q}_{obs}"

    if cfg.alpha_mode == "irrational_float":
        obs = cfg.observable_mode
        alpha_name = cfg.target_alpha_name or "irrational"
        return f"rotation_circle_irrational_{alpha_name}_{obs}"

    if cfg.alpha_mode == "irrational_approximant":
        obs = cfg.observable_mode
        alpha_name = cfg.target_alpha_name or "target"
        idx = cfg.approx_index if cfg.approx_index is not None else "na"
        return f"rotation_circle_approximant_{alpha_name}_{idx}_{obs}"

    return "rotation_run"


def _finalize_config(cfg: RotationRunConfig) -> RotationRunConfig:
    cfg2 = deepcopy(cfg)
    if not cfg2.run_name:
        cfg2.run_name = _default_run_name(cfg2)
    Path(cfg2.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg2


def execute_rotation_run(cfg: RotationRunConfig):
    cfg = _finalize_config(cfg)
    result = run_rotation_experiment(cfg)

    # reporting
    print_rotation_report(result)
    compare_peaks_to_expected_angles(result)

    # saving / plotting
    if cfg.save_npz:
        save_rotation_outputs(result)

    if cfg.make_plots:
        plot_rotation_result(result)

    return result


# ---------------------------------------------------------------------
# Optional smoke tests
# ---------------------------------------------------------------------

def run_smoke_tests() -> dict[str, object]:
    """Run the four baseline validation cases for first-pass integration."""
    out: dict[str, object] = {}

    # 1. Circle rational, eigenfunction
    cfg1 = deepcopy(CONFIG)
    cfg1.rotation_kind = "circle"
    cfg1.alpha_mode = "rational"
    cfg1.p = 1
    cfg1.q = 3
    cfg1.alpha = None
    cfg1.observable_mode = "eigenfunction"
    cfg1.observable_eigenvalue_index = 1
    cfg1.make_plots = False
    cfg1.save_npz = False
    cfg1.run_name = "smoke_circle_rational_1_3_eigenfunction"
    out["circle_rational_eigenfunction"] = execute_rotation_run(cfg1)

    # 2. Circle irrational, eigenfunction
    cfg2 = deepcopy(CONFIG)
    cfg2.rotation_kind = "circle"
    cfg2.alpha_mode = "irrational_float"
    cfg2.alpha = (5.0**0.5 - 1.0) / 2.0
    cfg2.observable_mode = "eigenfunction"
    cfg2.observable_eigenvalue_index = 1
    cfg2.make_plots = False
    cfg2.save_npz = False
    cfg2.run_name = "smoke_circle_irrational_eigenfunction"
    out["circle_irrational_eigenfunction"] = execute_rotation_run(cfg2)

    # 3. Circle irrational, rich observable
    cfg3 = deepcopy(CONFIG)
    cfg3.rotation_kind = "circle"
    cfg3.alpha_mode = "irrational_float"
    cfg3.alpha = (5.0**0.5 - 1.0) / 2.0
    cfg3.observable_mode = "rich"
    cfg3.make_plots = False
    cfg3.save_npz = False
    cfg3.run_name = "smoke_circle_irrational_rich"
    out["circle_irrational_rich"] = execute_rotation_run(cfg3)

    # 4. Planar rotation
    cfg4 = deepcopy(CONFIG)
    cfg4.rotation_kind = "planar"
    cfg4.theta = 2.0 * 3.141592653589793 * ((5.0**0.5 - 1.0) / 2.0)
    cfg4.alpha_mode = "irrational_float"   # ignored for planar
    cfg4.alpha = None
    cfg4.observable_mode = "eigenfunction"
    cfg4.make_plots = False
    cfg4.save_npz = False
    cfg4.run_name = "smoke_planar_eigenfunction"
    out["planar_eigenfunction"] = execute_rotation_run(cfg4)

    return out


if __name__ == "__main__":
    execute_rotation_run(CONFIG)