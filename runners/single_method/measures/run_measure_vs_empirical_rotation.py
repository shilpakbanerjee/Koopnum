"""
Compare exact and empirical moment sources for a pure point example.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from methods.common.moments.moment_sources import (
    ExactMomentSource,
    EmpiricalMomentSource,
)
from methods.cd_kernel.core.kernel import evaluate_cd_kernel_from_moments
from methods.cd_kernel.diagnostics.diagnostics import (
    summarize_result,
    compare_results,
    weak_convergence_summary,
)
from methods.common.systems import generate_planar_rotation
from methods.common.observables import complex_coordinate
from methods.common.plotting.common_plotting import (
    save_density_comparison_plot,
    save_density_comparison_log_plot,
    save_density_comparison_normalized_plot,
    save_peak_overlay_plot,
    save_difference_plot,
)


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/measures/measure_vs_empirical_rotation")
PLOT_DIR = Path("experiments/cd_kernel/plots/measures/measure_vs_empirical_rotation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def atomic_moment_function(theta: float):
    def mk(k: int) -> complex:
        return np.exp(-1j * k * theta)
    return mk


def main():
    theta = 0.35
    n = 2000
    order = 80
    grid_size = 2048
    regularization = 1e-6

    exact_source = ExactMomentSource(atomic_moment_function(theta))
    exact_moments = exact_source.moments(order)

    exact_result = evaluate_cd_kernel_from_moments(
        exact_moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=True,
    )

    X = generate_planar_rotation(n=n, theta=theta)
    signal = complex_coordinate(0, 1)(X)

    empirical_source = EmpiricalMomentSource(signal)
    empirical_moments = empirical_source.moments(order)

    empirical_result = evaluate_cd_kernel_from_moments(
        empirical_moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=True,
    )

    print("=== Exact vs empirical rotation comparison ===")
    print("theta =", theta)

    exact_summary = summarize_result(exact_result)
    empirical_summary = summarize_result(empirical_result)
    comparison = compare_results(exact_result, empirical_result)
    weak_summary = weak_convergence_summary(exact_result, empirical_result, max_mode=12)

    print("\nExact summary:")
    for k, v in exact_summary.items():
        print(f"  {k}: {v:.6e}")

    print("\nEmpirical summary:")
    for k, v in empirical_summary.items():
        print(f"  {k}: {v:.6e}")

    print("\nComparison diagnostics:")
    for k, v in comparison.items():
        print(f"  {k}: {v:.6e}")

    print("\nWeak-convergence diagnostics:")
    for k, v in weak_summary.items():
        print(f"  {k}: {v:.6e}")

    np.savez(
        OUTPUT_DIR / "measure_vs_empirical_rotation.npz",
        theta=theta,
        exact_moments=exact_moments,
        empirical_moments=empirical_moments,
        exact_angles=exact_result.angles,
        exact_density=exact_result.density_proxy,
        exact_kernel=exact_result.kernel_diag,
        empirical_angles=empirical_result.angles,
        empirical_density=empirical_result.density_proxy,
        empirical_kernel=empirical_result.kernel_diag,
        exact_summary=np.array(list(exact_summary.items()), dtype=object),
        empirical_summary=np.array(list(empirical_summary.items()), dtype=object),
        comparison=np.array(list(comparison.items()), dtype=object),
        weak_summary=np.array(list(weak_summary.items()), dtype=object),
    )

    styles = [
        {"linestyle": "--", "alpha": 0.75, "linewidth": 1.8},
        {"linestyle": "-", "alpha": 0.95, "linewidth": 1.6},
    ]

    save_density_comparison_plot(
        results=[empirical_result, exact_result],
        labels=["Empirical moments", "Exact moments"],
        title="Rotation: exact vs empirical CD reconstruction",
        save_path=PLOT_DIR / "rotation_exact_vs_empirical_density.png",
        show_peaks=True,
        styles=styles,
    )

    save_density_comparison_log_plot(
        results=[empirical_result, exact_result],
        labels=["Empirical moments", "Exact moments"],
        title="Rotation: exact vs empirical (log scale)",
        save_path=PLOT_DIR / "rotation_exact_vs_empirical_density_log.png",
        styles=styles,
    )

    save_density_comparison_normalized_plot(
        results=[empirical_result, exact_result],
        labels=["Empirical / max", "Exact / max"],
        title="Rotation: exact vs empirical (normalized)",
        save_path=PLOT_DIR / "rotation_exact_vs_empirical_density_normalized.png",
        styles=styles,
    )

    save_peak_overlay_plot(
        base_result=exact_result,
        overlay_results=[exact_result, empirical_result],
        overlay_labels=["Exact peaks", "Empirical peaks"],
        title="Rotation: exact vs empirical peak locations",
        save_path=PLOT_DIR / "rotation_exact_vs_empirical_peaks.png",
        base_label="Exact density",
    )

    save_difference_plot(
        result_a=exact_result,
        result_b=empirical_result,
        title="Rotation: |exact density - empirical density|",
        save_path=PLOT_DIR / "rotation_exact_vs_empirical_difference.png",
    )

    plt.show()


if __name__ == "__main__":
    main()