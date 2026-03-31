from __future__ import annotations

"""Comparison runner for rotation experiments."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any

import csv
import numpy as np
import matplotlib.pyplot as plt

from runners.single_method.dynamics._rotation_config import RotationRunConfig
from runners.single_method.dynamics.run_rotation import execute_rotation_run

from methods.cd_kernel.core.weak_convergence import (
    compare_rotation_measures_weakly,
    smoothed_density_on_grid,
)


CompareMode = Literal[
    "irrational_vs_approximants",
    "observable_modes",
    "rational_family",
]


@dataclass
class RotationComparisonConfig:
    compare_mode: CompareMode = "irrational_vs_approximants"
    base_config: RotationRunConfig = field(default_factory=RotationRunConfig)
    approx_indices: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    observable_modes: list[str] = field(default_factory=lambda: ["eigenfunction", "rich"])
    rational_pairs: list[tuple[int, int]] = field(default_factory=lambda: [(1, 2), (2, 3), (3, 5), (5, 8)])
    output_dir: str = "outputs/rotation_comparison"
    make_plots: bool = True
    save_csv: bool = True


def _circular_distance(a: float, b: float) -> float:
    d = abs(a - b) % (2.0 * np.pi)
    return min(d, 2.0 * np.pi - d)


def _target_alpha_from_base(cfg: RotationRunConfig) -> tuple[float, str]:
    if cfg.alpha is not None:
        return float(cfg.alpha), (cfg.target_alpha_name or "target_alpha")
    if cfg.target_alpha is not None:
        return float(cfg.target_alpha), (cfg.target_alpha_name or "target_alpha")
    raise ValueError("Base config must define alpha or target_alpha for irrational_vs_approximants mode.")


def _continued_fraction_convergents(alpha: float, n_terms: int) -> list[tuple[int, int]]:
    x = float(alpha)
    a = []
    for _ in range(max(1, n_terms)):
        ai = int(np.floor(x))
        a.append(ai)
        frac = x - ai
        if abs(frac) < 1e-14:
            break
        x = 1.0 / frac

    convs: list[tuple[int, int]] = []
    p_nm2, p_nm1 = 0, 1
    q_nm2, q_nm1 = 1, 0
    for ai in a:
        p_n = ai * p_nm1 + p_nm2
        q_n = ai * q_nm1 + q_nm2
        convs.append((p_n, q_n))
        p_nm2, p_nm1 = p_nm1, p_n
        q_nm2, q_nm1 = q_nm1, q_n
    return convs


def _build_single_approximant_cfg(base: RotationRunConfig, approx_index: int) -> RotationRunConfig:
    cfg = deepcopy(base)
    target_alpha, target_name = _target_alpha_from_base(base)
    convs = _continued_fraction_convergents(target_alpha, max(approx_index + 2, 8))
    if approx_index < 0 or approx_index >= len(convs):
        raise ValueError(f"approx_index={approx_index} is out of range for available convergents")

    p, q = convs[approx_index]
    cfg.rotation_kind = "circle"
    cfg.alpha_mode = "rational"
    cfg.alpha = None
    cfg.p = p
    cfg.q = q
    cfg.target_alpha = target_alpha
    cfg.target_alpha_name = target_name
    cfg.run_name = f"{cfg.observable_mode}_approx_{approx_index}_{p}_{q}"
    return cfg


def _build_direct_irrational_cfg(base: RotationRunConfig) -> RotationRunConfig:
    cfg = deepcopy(base)
    target_alpha, target_name = _target_alpha_from_base(base)
    cfg.rotation_kind = "circle"
    cfg.alpha_mode = "irrational_float"
    cfg.alpha = target_alpha
    cfg.target_alpha_name = target_name
    cfg.run_name = f"{cfg.observable_mode}_irrational_direct"
    return cfg


def _write_summary_csv(path, rows):
    import csv

    if not rows:
        return

    # collect all keys across rows (important: rows may have different keys)
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = sorted(all_keys)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for row in rows:
            # fill missing keys with None
            full_row = {k: row.get(k, None) for k in fieldnames}
            writer.writerow(full_row)


def _summary_row(result: Any, label: str, alpha_target: float | None = None) -> dict[str, Any]:
    peaks = result.detected_peaks
    top_peak_angle = float(peaks[0]["angle"]) if peaks else np.nan
    top_peak_value = float(peaks[0]["value"]) if peaks else np.nan

    row = {
        "label": label,
        "case_name": result.resolved.case_name,
        "observable_mode": result.config.observable_mode,
        "alpha_effective": getattr(result.resolved, "alpha_effective", np.nan),
        "theta_effective": getattr(result.resolved, "theta_effective", np.nan),
        "toeplitz_condition_number": result.cd_result.metadata.get("gram_condition_number", np.nan),
        "top_peak_angle": top_peak_angle,
        "top_peak_value": top_peak_value,
    }

    if alpha_target is not None and getattr(result.resolved, "alpha_effective", None) is not None:
        row["alpha_error"] = abs(float(result.resolved.alpha_effective) - float(alpha_target))

    if len(result.observable.expected_angles) > 0 and peaks:
        row["nearest_expected_error"] = min(
            _circular_distance(float(a), top_peak_angle) for a in result.observable.expected_angles
        )
    else:
        row["nearest_expected_error"] = np.nan

    return row


def _plot_atomic_overlay(results: list[Any], outdir: Path, filename: str, title: str) -> str:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    for result in results:
        angles = np.asarray(result.cd_result.angles, dtype=float)
        atomic_proxy = np.asarray(result.atomic_proxy, dtype=float)
        label = result.config.run_name or "run"
        ax.plot(angles, atomic_proxy, linewidth=1.4, label=label)

    ax.set_xlabel("angle")
    ax.set_ylabel("atomic proxy")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_condition_vs_denominator(rows: list[dict[str, Any]], outdir: Path, filename: str) -> str:
    xs = []
    ys = []
    for row in rows:
        label = str(row["label"])
        if "/" in label:
            try:
                q = int(label.split("/")[1])
                xs.append(q)
                ys.append(float(row["toeplitz_condition_number"]))
            except Exception:
                pass

    if not xs:
        return ""

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("denominator q")
    ax.set_ylabel("toeplitz condition number")
    ax.set_title("Condition number vs denominator")
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_rotation_comparison(cfg: RotationComparisonConfig) -> dict[str, Any]:
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[Any] = []
    rows: list[dict[str, Any]] = []
    files: dict[str, str] = {}

    if cfg.compare_mode == "irrational_vs_approximants":
        base = deepcopy(cfg.base_config)
        base.output_dir = str(outdir / "single_runs")
        alpha_target, _ = _target_alpha_from_base(base)

        direct_cfg = _build_direct_irrational_cfg(base)
        direct_cfg.make_plots = False
        direct_cfg.save_npz = True
        direct_result = execute_rotation_run(direct_cfg)
        results.append(direct_result)
        rows.append(_summary_row(direct_result, label="direct", alpha_target=alpha_target))

        for idx in cfg.approx_indices:
            approx_cfg = _build_single_approximant_cfg(base, idx)
            approx_cfg.output_dir = str(outdir / "single_runs")
            approx_cfg.make_plots = False
            approx_cfg.save_npz = True
            result = execute_rotation_run(approx_cfg)
            results.append(result)
            label = f"{approx_cfg.p}/{approx_cfg.q}"
            rows.append(_summary_row(result, label=label, alpha_target=alpha_target))

        if cfg.save_csv:
            csv_path = outdir / "irrational_vs_approximants_summary.csv"
            _write_summary_csv(csv_path, rows)
            files["summary_csv"] = str(csv_path)

        if cfg.make_plots:
            overlay_path = _plot_atomic_overlay(
                results,
                outdir,
                "irrational_vs_approximants_atomic_overlay.png",
                "Atomic proxy: direct irrational vs rational approximants",
            )
            files["atomic_overlay"] = overlay_path

            cond_path = _plot_condition_vs_denominator(
                rows,
                outdir,
                "irrational_vs_approximants_condition_vs_denominator.png",
            )
            if cond_path:
                files["condition_vs_denominator"] = cond_path

            # Weak-convergence diagnostics from the exact atomic measures
            weak = compare_rotation_measures_weakly(
                target_alpha=alpha_target,
                approximants=[(int(r["p"]), int(r["q"])) for r in [
                    {"p": result.config.p, "q": result.config.q}
                    for result in results[1:]  # skip direct irrational result
                ]],
                harmonics=direct_result.observable.harmonics,
                coefficients=direct_result.observable.coefficients,
                max_moment_order=5,
                bump_sigma=0.25,
            )

            weak_rows = weak["rows"]

            if cfg.save_csv:
                weak_csv_path = outdir / "irrational_vs_approximants_weak_summary.csv"
                _write_summary_csv(weak_csv_path, weak_rows)
                files["weak_summary_csv"] = str(weak_csv_path)

            if cfg.make_plots and weak_rows:
                weak_moment_path = _plot_weak_moment_errors(
                    weak_rows,
                    outdir,
                    "irrational_vs_approximants_weak_moment_errors.png",
                )
                files["weak_moment_errors"] = weak_moment_path

                weak_test_path = _plot_weak_test_errors(
                    weak_rows,
                    outdir,
                    "irrational_vs_approximants_weak_test_errors.png",
                )
                files["weak_test_errors"] = weak_test_path

                smoothed_overlay_path = _plot_smoothed_measure_overlay(
                    target_alpha=alpha_target,
                    approximants=[(int(r["p"]), int(r["q"])) for r in weak_rows],
                    harmonics=direct_result.observable.harmonics,
                    coefficients=direct_result.observable.coefficients,
                    outdir=outdir,
                    filename="irrational_vs_approximants_smoothed_overlay.png",
                    sigma=0.20,
                )
                files["smoothed_overlay"] = smoothed_overlay_path

        return {"results": results, "rows": rows, "files": files}

    if cfg.compare_mode == "observable_modes":
        base = deepcopy(cfg.base_config)
        base.output_dir = str(outdir / "single_runs")

        for mode in cfg.observable_modes:
            run_cfg = deepcopy(base)
            run_cfg.observable_mode = mode
            run_cfg.run_name = f"observable_mode_{mode}"
            run_cfg.make_plots = False
            run_cfg.save_npz = True
            result = execute_rotation_run(run_cfg)
            results.append(result)
            rows.append(_summary_row(result, label=mode))

        if cfg.save_csv:
            csv_path = outdir / "observable_modes_summary.csv"
            _write_summary_csv(csv_path, rows)
            files["summary_csv"] = str(csv_path)

        if cfg.make_plots:
            overlay_path = _plot_atomic_overlay(
                results,
                outdir,
                "observable_modes_atomic_overlay.png",
                "Atomic proxy across observable modes",
            )
            files["atomic_overlay"] = overlay_path

        return {"results": results, "rows": rows, "files": files}

    if cfg.compare_mode == "rational_family":
        base = deepcopy(cfg.base_config)
        base.output_dir = str(outdir / "single_runs")

        for p, q in cfg.rational_pairs:
            run_cfg = deepcopy(base)
            run_cfg.rotation_kind = "circle"
            run_cfg.alpha_mode = "rational"
            run_cfg.alpha = None
            run_cfg.p = p
            run_cfg.q = q
            run_cfg.run_name = f"rational_{p}_{q}"
            run_cfg.make_plots = False
            run_cfg.save_npz = True
            result = execute_rotation_run(run_cfg)
            results.append(result)
            rows.append(_summary_row(result, label=f"{p}/{q}"))

        if cfg.save_csv:
            csv_path = outdir / "rational_family_summary.csv"
            _write_summary_csv(csv_path, rows)
            files["summary_csv"] = str(csv_path)

        if cfg.make_plots:
            overlay_path = _plot_atomic_overlay(
                results,
                outdir,
                "rational_family_atomic_overlay.png",
                "Atomic proxy across rational rotation family",
            )
            files["atomic_overlay"] = overlay_path

            cond_path = _plot_condition_vs_denominator(
                rows,
                outdir,
                "rational_family_condition_vs_denominator.png",
            )
            if cond_path:
                files["condition_vs_denominator"] = cond_path

        return {"results": results, "rows": rows, "files": files}

    raise ValueError(f"Unsupported compare_mode={cfg.compare_mode!r}")

def _plot_weak_moment_errors(rows: list[dict[str, Any]], outdir: Path, filename: str) -> str:
    q_vals = []
    series: dict[str, list[float]] = {}

    moment_keys = [k for k in rows[0].keys() if k.startswith("moment_") and k.endswith("_abs_error")]
    moment_keys.sort()

    for key in moment_keys:
        series[key] = []

    for row in rows:
        q_vals.append(int(row["q"]))
        for key in moment_keys:
            series[key].append(float(row[key]))

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    for key in moment_keys:
        ax.plot(q_vals, series[key], marker="o", label=key)
    ax.set_xlabel("denominator q")
    ax.set_ylabel("absolute error")
    ax.set_title("Weak convergence via moment errors")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_weak_test_errors(rows: list[dict[str, Any]], outdir: Path, filename: str) -> str:
    q_vals = []
    series: dict[str, list[float]] = {}

    test_keys = [k for k in rows[0].keys() if k.startswith("test_") and k.endswith("_abs_error")]
    test_keys.sort()

    # Keep the plot readable on first pass
    preferred = [k for k in test_keys if ("cos_" in k or "sin_" in k or "bump_atom_1" in k or "bump_pi" in k)]
    chosen = preferred[:6] if preferred else test_keys[:6]

    for key in chosen:
        series[key] = []

    for row in rows:
        q_vals.append(int(row["q"]))
        for key in chosen:
            series[key].append(float(row[key]))

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    for key in chosen:
        ax.plot(q_vals, series[key], marker="o", label=key)
    ax.set_xlabel("denominator q")
    ax.set_ylabel("absolute error")
    ax.set_title("Weak convergence via continuous test functions")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_smoothed_measure_overlay(
    target_alpha: float,
    approximants: list[tuple[int, int]],
    harmonics: np.ndarray,
    coefficients: np.ndarray,
    outdir: Path,
    filename: str,
    sigma: float = 0.20,
) -> str:
    from methods.cd_kernel.core.weak_convergence import build_rotation_measure

    angle_grid = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    mu_target = build_rotation_measure(target_alpha, harmonics, coefficients)
    y_target = smoothed_density_on_grid(mu_target, angle_grid, sigma=sigma)
    ax.plot(angle_grid, y_target, linewidth=2.0, label="target irrational")

    for p, q in approximants:
        mu_n = build_rotation_measure(float(p) / float(q), harmonics, coefficients)
        y_n = smoothed_density_on_grid(mu_n, angle_grid, sigma=sigma)
        ax.plot(angle_grid, y_n, linewidth=1.2, label=f"{p}/{q}")

    ax.set_xlabel("angle")
    ax.set_ylabel("smoothed mass profile")
    ax.set_title("Weak convergence view: smoothed spectral measures")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------
# First-pass default config
# ---------------------------------------------------------------------

BASE_CONFIG = RotationRunConfig(
    rotation_kind="circle",
    alpha_mode="irrational_float",
    alpha=(5.0**0.5 - 1.0) / 2.0,
    observable_mode="rich",
    observable_eigenvalue_index=1,
    x0=0.123456789,
    n_traj=4000,
    moment_order=120,
    grid_size=512,
    regularization=1e-8,
    koopman_mode="none",
    koopman_order=30,
    output_dir="outputs/rotation_comparison",
    run_name=None,
    make_plots=False,
    save_npz=True,
)

CONFIG = RotationComparisonConfig(
    compare_mode="irrational_vs_approximants",
    base_config=BASE_CONFIG,
    approx_indices=[2, 3, 4, 5, 6],
    output_dir="outputs/rotation_comparison",
    make_plots=True,
    save_csv=True,
)


if __name__ == "__main__":
    run_rotation_comparison(CONFIG)