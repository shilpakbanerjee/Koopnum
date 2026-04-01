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


# ---------------------------------------------------------------------
# Formatting / labels
# ---------------------------------------------------------------------

def _alpha_display(target_alpha: float, alpha_name: str | None = None) -> str:
    if alpha_name == "golden_conjugate":
        return rf"$\alpha=\frac{{\sqrt{{5}}-1}}{{2}}\approx {target_alpha:.6f}$"
    if alpha_name == "sqrt2_minus_1":
        return rf"$\alpha=\sqrt{{2}}-1\approx {target_alpha:.6f}$"
    if alpha_name == "sqrt3_minus_1":
        return rf"$\alpha=\sqrt{{3}}-1\approx {target_alpha:.6f}$"
    return rf"$\alpha\approx {target_alpha:.6f}$"


def _approximant_label(result) -> str:
    if getattr(result.config, "p", None) is not None and getattr(result.config, "q", None) is not None:
        return f"approximant {result.config.p}/{result.config.q}"
    return "irrational target"


def _test_function_pretty_name(key: str) -> str:
    mapping = {
        "test_bump_atom_1_abs_error": "bump at atom 1",
        "test_bump_atom_2_abs_error": "bump at atom 2",
        "test_bump_atom_3_abs_error": "bump at atom 3",
        "test_bump_pi_abs_error": "bump at π",
        "test_cos_1_abs_error": "cos(θ)",
        "test_cos_2_abs_error": "cos(2θ)",
        "test_cos_3_abs_error": "cos(3θ)",
        "test_sin_1_abs_error": "sin(θ)",
        "test_sin_2_abs_error": "sin(2θ)",
        "test_sin_3_abs_error": "sin(3θ)",
    }
    return mapping.get(key, key)


def _moment_pretty_name(key: str) -> str:
    parts = key.split("_")
    if len(parts) >= 2:
        k = parts[1]
        return rf"$|m_{{{k}}}^{{(p/q)}}-m_{{{k}}}^{{(\alpha)}}|$"
    return key


def _header_lines(
    title: str,
    target_alpha: float | None = None,
    observable_desc: str | None = None,
    alpha_name: str | None = None,
    extra_lines: list[str] | None = None,
) -> list[str]:
    lines = [title]
    if target_alpha is not None:
        lines.append("target rotation: " + _alpha_display(target_alpha, alpha_name=alpha_name))
    if observable_desc is not None:
        lines.append(rf"observable: ${observable_desc}$")
    if extra_lines:
        lines.extend(extra_lines)
    return lines


def _apply_figure_header(
    fig,
    ax,
    title: str,
    target_alpha: float | None = None,
    observable_desc: str | None = None,
    alpha_name: str | None = None,
    extra_lines: list[str] | None = None,
) -> None:
    """
    Draw a clean multi-line header with explicit vertical spacing.

    Layout strategy:
    - all header text is figure-level
    - smaller fonts than before
    - axes top is computed from number of lines
    - no bbox_inches='tight' when saving
    """
    lines = _header_lines(
        title=title,
        target_alpha=target_alpha,
        observable_desc=observable_desc,
        alpha_name=alpha_name,
        extra_lines=extra_lines,
    )

    title_fs = 18
    meta_fs = 11

    # Vertical positions in figure coordinates
    y_top = 0.972
    dy = 0.045

    for i, line in enumerate(lines):
        y = y_top - i * dy
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            va="top",
            fontsize=(title_fs if i == 0 else meta_fs),
        )

    # Compute top of axes from number of header lines.
    # More lines => slightly lower axes, but without huge empty gap.
    n = len(lines)
    top = 0.88 - n * 0.045
    top = max(0.72, min(0.80, top))

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=top)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

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
    if not rows:
        return

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = sorted(all_keys)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
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
        "toeplitz_condition_number": result.cd_result.metadata.get("toeplitz_condition_number", np.nan),
        "top_peak_angle": top_peak_angle,
        "top_peak_value": top_peak_value,
    }

    if getattr(result.config, "p", None) is not None:
        row["p"] = int(result.config.p)
    if getattr(result.config, "q", None) is not None:
        row["q"] = int(result.config.q)

    if alpha_target is not None and getattr(result.resolved, "alpha_effective", None) is not None:
        row["alpha_error"] = abs(float(result.resolved.alpha_effective) - float(alpha_target))

    if len(result.observable.expected_angles) > 0 and peaks:
        row["nearest_expected_error"] = min(
            _circular_distance(float(a), top_peak_angle) for a in result.observable.expected_angles
        )
    else:
        row["nearest_expected_error"] = np.nan

    return row


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def _plot_atomic_overlay(results, outdir, filename, title, target_alpha, observable_desc, alpha_name=None):
    fig, ax = plt.subplots(figsize=(10.8, 6.0))

    for result in results:
        angles = np.asarray(result.cd_result.angles, dtype=float)
        atomic_proxy = np.asarray(result.atomic_proxy, dtype=float)
        ax.plot(angles, atomic_proxy, linewidth=1.6, label=_approximant_label(result))

    ax.set_xlabel("angle on unit circle (radians)")
    ax.set_ylabel("atomic spectral proxy")
    ax.legend(loc="best", fontsize=10, frameon=True)

    _apply_figure_header(fig, ax, title, target_alpha, observable_desc, alpha_name=alpha_name)

    path = outdir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _plot_condition_vs_denominator(rows, outdir, filename, target_alpha=None, observable_desc=None, alpha_name=None):
    pairs = []

    for row in rows:
        if "q" not in row or "toeplitz_condition_number" not in row:
            continue
        try:
            q_val = int(row["q"])
            cond_val = float(row["toeplitz_condition_number"])
        except Exception:
            continue
        if np.isfinite(q_val) and np.isfinite(cond_val):
            pairs.append((q_val, cond_val))

    if not pairs:
        return ""

    pairs = sorted(pairs, key=lambda t: t[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.plot(xs, ys, marker="o", linewidth=1.8)
    ax.set_xlabel("denominator q of periodic approximant")
    ax.set_ylabel("Toeplitz condition number")

    _apply_figure_header(
        fig,
        ax,
        "Toeplitz condition number vs approximant denominator",
        target_alpha,
        observable_desc,
        alpha_name=alpha_name,
    )

    path = outdir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _plot_weak_moment_errors(rows, outdir, filename, target_alpha, observable_desc, alpha_name=None):
    q_vals = []
    series = {}

    moment_keys = [k for k in rows[0].keys() if k.startswith("moment_") and k.endswith("_abs_error")]
    moment_keys.sort()

    for key in moment_keys:
        series[key] = []

    for row in rows:
        q_vals.append(int(row["q"]))
        for key in moment_keys:
            series[key].append(float(row[key]))

    fig, ax = plt.subplots(figsize=(10.6, 6.0))

    for key in moment_keys:
        ax.plot(q_vals, series[key], marker="o", label=_moment_pretty_name(key))

    ax.set_xlabel("denominator q of periodic approximant")
    ax.set_ylabel("absolute error")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=10, frameon=True)

    _apply_figure_header(
        fig,
        ax,
        "Weak convergence via moment errors",
        target_alpha,
        observable_desc,
        alpha_name=alpha_name,
        extra_lines=[r"moments shown: $k=1,\dots,5$"],
    )

    path = outdir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _plot_weak_test_errors(rows, outdir, filename, target_alpha, observable_desc, alpha_name=None):
    q_vals = []
    series = {}

    test_keys = [k for k in rows[0].keys() if k.startswith("test_") and k.endswith("_abs_error")]
    test_keys.sort()

    preferred = [k for k in test_keys if ("cos_" in k or "sin_" in k or "bump_atom_1" in k or "bump_pi" in k)]
    chosen = preferred[:6] if preferred else test_keys[:6]

    for key in chosen:
        series[key] = []

    for row in rows:
        q_vals.append(int(row["q"]))
        for key in chosen:
            series[key].append(float(row[key]))

    fig, ax = plt.subplots(figsize=(10.6, 5.9))

    for key in chosen:
        ax.plot(q_vals, series[key], marker="o", label=_test_function_pretty_name(key))

    ax.set_xlabel("denominator q of periodic approximant")
    ax.set_ylabel("absolute error")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=10, frameon=True)

    _apply_figure_header(
        fig,
        ax,
        "Weak convergence via continuous test functions",
        target_alpha,
        observable_desc,
        alpha_name=alpha_name,
    )

    path = outdir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _plot_smoothed_measure_overlay(
    target_alpha,
    approximants,
    harmonics,
    coefficients,
    outdir,
    filename,
    observable_desc,
    alpha_name=None,
    sigma: float = 0.20,
) -> str:
    from methods.cd_kernel.core.weak_convergence import build_rotation_measure

    angle_grid = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)

    fig, ax = plt.subplots(figsize=(10.8, 6.0))

    mu_target = build_rotation_measure(target_alpha, harmonics, coefficients)
    y_target = smoothed_density_on_grid(mu_target, angle_grid, sigma=sigma)
    ax.plot(angle_grid, y_target, linewidth=2.0, label="irrational target")

    for p, q in approximants:
        mu_n = build_rotation_measure(float(p) / float(q), harmonics, coefficients)
        y_n = smoothed_density_on_grid(mu_n, angle_grid, sigma=sigma)
        ax.plot(angle_grid, y_n, linewidth=1.3, label=f"approximant {p}/{q}")

    ax.set_xlabel("angle on unit circle (radians)")
    ax.set_ylabel("smoothed spectral mass")
    ax.legend(loc="best", fontsize=10, frameon=True)

    _apply_figure_header(
        fig,
        ax,
        "Weak convergence view through smoothed spectral measures",
        target_alpha,
        observable_desc,
        alpha_name=alpha_name,
        extra_lines=[rf"smoothing parameter: $\sigma={sigma:.2f}$"],
    )

    path = outdir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _plot_collapse_metrics(rows, outdir, filename, target_alpha, observable_desc, alpha_name=None):
    q_vals = [int(row["q"]) for row in rows]
    entropy = [float(row["spectral_entropy"]) for row in rows]
    eff_atoms = [float(row["effective_atom_count"]) for row in rows]
    top1 = [float(row["top_1_mass"]) for row in rows]
    ratio = [float(row["concentration_ratio"]) for row in rows]
    num_atoms = [float(row["num_atoms"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10.4, 5.9))

    ax.plot(q_vals, entropy, marker="o", label="entropy")
    ax.plot(q_vals, eff_atoms, marker="s", label="effective atom count")
    ax.plot(q_vals, top1, marker="^", label="largest atom mass")
    ax.plot(q_vals, num_atoms, marker="d", label="number of atoms")

    finite_q = [q for q, r in zip(q_vals, ratio) if np.isfinite(r)]
    finite_r = [r for r in ratio if np.isfinite(r)]
    if finite_q:
        ax.plot(finite_q, finite_r, marker="x", label="largest / second-largest mass")

    ax.set_xlabel("denominator q of periodic approximant")
    ax.set_ylabel("metric value")
    ax.legend(loc="best", fontsize=10, frameon=True)

    _apply_figure_header(
        fig,
        ax,
        "Spectral collapse diagnostics vs approximant denominator",
        target_alpha,
        observable_desc,
        alpha_name=alpha_name,
    )

    path = outdir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _print_collapse_summary(rows):
    print("\n=== Collapse diagnostics summary ===")
    for row in rows:
        q = row.get("q", "NA")
        num_atoms = row.get("num_atoms", "NA")
        collisions = row.get("collision_count", "NA")
        entropy = row.get("spectral_entropy", float("nan"))
        eff_atoms = row.get("effective_atom_count", float("nan"))
        top1 = row.get("top_1_mass", float("nan"))
        top2 = row.get("top_2_mass", float("nan"))
        ratio = row.get("concentration_ratio", float("nan"))

        print(
            f"q={q:>3} | "
            f"num_atoms={num_atoms} | "
            f"collisions={collisions} | "
            f"entropy={entropy:.6f} | "
            f"effective_atoms={eff_atoms:.6f} | "
            f"top1={top1:.6f} | "
            f"top2={top2:.6f} | "
            f"ratio={ratio}"
        )


# ---------------------------------------------------------------------
# Main comparison driver
# ---------------------------------------------------------------------

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

        observable_desc = direct_result.observable.description
        alpha_name = getattr(direct_cfg, "target_alpha_name", None)

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
                "Atomic spectral proxy under periodic approximation",
                target_alpha=alpha_target,
                observable_desc=observable_desc,
                alpha_name=alpha_name,
            )
            files["atomic_overlay"] = overlay_path

            cond_path = _plot_condition_vs_denominator(
                rows,
                outdir,
                "irrational_vs_approximants_condition_vs_denominator.png",
                target_alpha=alpha_target,
                observable_desc=observable_desc,
                alpha_name=alpha_name,
            )
            if cond_path:
                files["condition_vs_denominator"] = cond_path

            weak = compare_rotation_measures_weakly(
                target_alpha=alpha_target,
                approximants=[(int(result.config.p), int(result.config.q)) for result in results[1:]],
                harmonics=direct_result.observable.harmonics,
                coefficients=direct_result.observable.coefficients,
                max_moment_order=5,
                bump_sigma=0.25,
            )

            weak_rows = weak["rows"]
            _print_collapse_summary(weak_rows)

            if cfg.save_csv:
                weak_csv_path = outdir / "irrational_vs_approximants_weak_summary.csv"
                _write_summary_csv(weak_csv_path, weak_rows)
                files["weak_summary_csv"] = str(weak_csv_path)

            collapse_plot_path = _plot_collapse_metrics(
                weak_rows,
                outdir,
                "irrational_vs_approximants_collapse_metrics.png",
                target_alpha=alpha_target,
                observable_desc=observable_desc,
                alpha_name=alpha_name,
            )
            files["collapse_metrics"] = collapse_plot_path

            if weak_rows:
                weak_moment_path = _plot_weak_moment_errors(
                    weak_rows,
                    outdir,
                    "irrational_vs_approximants_weak_moment_errors.png",
                    target_alpha=alpha_target,
                    observable_desc=observable_desc,
                    alpha_name=alpha_name,
                )
                files["weak_moment_errors"] = weak_moment_path

                weak_test_path = _plot_weak_test_errors(
                    weak_rows,
                    outdir,
                    "irrational_vs_approximants_weak_test_errors.png",
                    target_alpha=alpha_target,
                    observable_desc=observable_desc,
                    alpha_name=alpha_name,
                )
                files["weak_test_errors"] = weak_test_path

                smoothed_overlay_path = _plot_smoothed_measure_overlay(
                    target_alpha=alpha_target,
                    approximants=[(int(r["p"]), int(r["q"])) for r in weak_rows],
                    harmonics=direct_result.observable.harmonics,
                    coefficients=direct_result.observable.coefficients,
                    outdir=outdir,
                    filename="irrational_vs_approximants_smoothed_overlay.png",
                    observable_desc=observable_desc,
                    alpha_name=alpha_name,
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
                target_alpha=None,
                observable_desc=None,
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
                target_alpha=None,
                observable_desc=None,
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


# ---------------------------------------------------------------------
# Default config
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