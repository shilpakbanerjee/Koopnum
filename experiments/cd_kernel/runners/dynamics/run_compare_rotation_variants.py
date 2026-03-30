"""
Compare baseline and tapered CD-kernel variants on planar rotation.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from methods.cd_kernel.diagnostics.diagnostics import (
    summarize_result,
    compare_results,
    weak_convergence_summary,
)
from methods.common.systems import generate_planar_rotation
from methods.common.observables import complex_coordinate
from methods.cd_kernel.dynamics.spectral_measure import (
    reconstruct_spectral_measure_from_system,
)
from methods.cd_kernel.core.cd_kernel_v002_tapered import (
    fit_cd_kernel_tapered_from_moments,
)
from methods.common.plotting.common_plotting import (
    save_density_comparison_plot,
    save_density_comparison_log_plot,
    save_density_comparison_normalized_plot,
    save_kernel_comparison_plot,
    save_peak_overlay_plot,
    save_unit_circle_peaks_plot,
    save_difference_plot,
)


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/dynamics/rotation_compare")
PLOT_DIR = Path("experiments/cd_kernel/plots/dynamics/rotation_compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    n = 2000
    theta = 0.35
    order = 80
    grid_size = 2048
    regularization = 1e-6

    observable = complex_coordinate(0, 1)

    X, baseline_spec, baseline = reconstruct_spectral_measure_from_system(
        system_fn=generate_planar_rotation,
        system_kwargs={"n": n, "theta": theta},
        order=order,
        observable=observable,
        center=False,
        normalize_moments=True,
        taper=None,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=True,
    )
    baseline.metadata["variant"] = "cd_kernel_v001_baseline"

    tapered = fit_cd_kernel_tapered_from_moments(
        moments=baseline_spec.moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        taper="fejer",
        normalize_density=True,
    )
    tapered.metadata["variant"] = "cd_kernel_v002_tapered"

    print("=== Rotation: baseline vs tapered ===")
    print("theta =", theta)

    baseline_summary = summarize_result(baseline)
    tapered_summary = summarize_result(tapered)
    comparison = compare_results(baseline, tapered)
    weak_summary = weak_convergence_summary(baseline, tapered, max_mode=12)

    print("\nBaseline summary:")
    for k, v in baseline_summary.items():
        print(f"  {k}: {v:.6e}")

    print("\nTapered summary:")
    for k, v in tapered_summary.items():
        print(f"  {k}: {v:.6e}")

    print("\nComparison diagnostics:")
    for k, v in comparison.items():
        print(f"  {k}: {v:.6e}")

    print("\nWeak-convergence diagnostics:")
    for k, v in weak_summary.items():
        print(f"  {k}: {v:.6e}")

    np.savez(
        OUTPUT_DIR / "rotation_compare_results.npz",
        trajectory=X,
        signal=baseline_spec.signal,
        moments=baseline_spec.moments,
        baseline_angles=baseline.angles,
        baseline_density=baseline.density_proxy,
        baseline_kernel=baseline.kernel_diag,
        tapered_angles=tapered.angles,
        tapered_density=tapered.density_proxy,
        tapered_kernel=tapered.kernel_diag,
        baseline_summary=np.array(list(baseline_summary.items()), dtype=object),
        tapered_summary=np.array(list(tapered_summary.items()), dtype=object),
        comparison=np.array(list(comparison.items()), dtype=object),
        weak_summary=np.array(list(weak_summary.items()), dtype=object),
    )

    styles = [
        {"linestyle": "-", "alpha": 0.85, "linewidth": 1.6},
        {"linestyle": "--", "alpha": 0.90, "linewidth": 1.6},
    ]

    save_density_comparison_plot(
        results=[baseline, tapered],
        labels=["Baseline", "Tapered (Fejér)"],
        title="Planar rotation: baseline vs tapered CD kernel",
        save_path=PLOT_DIR / "rotation_compare_density.png",
        show_peaks=True,
        styles=styles,
    )

    save_density_comparison_log_plot(
        results=[baseline, tapered],
        labels=["Baseline", "Tapered (Fejér)"],
        title="Planar rotation: density comparison (log scale)",
        save_path=PLOT_DIR / "rotation_compare_density_log.png",
        styles=styles,
    )

    save_density_comparison_normalized_plot(
        results=[baseline, tapered],
        labels=["Baseline / max", "Tapered / max"],
        title="Planar rotation: max-normalized comparison",
        save_path=PLOT_DIR / "rotation_compare_density_normalized.png",
        styles=styles,
    )

    save_kernel_comparison_plot(
        results=[baseline, tapered],
        labels=["Baseline kernel", "Tapered kernel"],
        title="Kernel diagonal: baseline vs tapered",
        save_path=PLOT_DIR / "rotation_compare_kernel.png",
        styles=styles,
    )

    save_peak_overlay_plot(
        base_result=tapered,
        overlay_results=[baseline, tapered],
        overlay_labels=["Baseline peaks", "Tapered peaks"],
        title="Planar rotation: detected peaks",
        save_path=PLOT_DIR / "rotation_compare_peaks.png",
        base_label="Tapered density",
    )

    save_unit_circle_peaks_plot(
        result=tapered,
        title="Planar rotation: tapered peaks on the unit circle",
        save_path=PLOT_DIR / "rotation_unit_circle.png",
    )

    save_difference_plot(
        result_a=baseline,
        result_b=tapered,
        title="Planar rotation: |baseline density - tapered density|",
        save_path=PLOT_DIR / "rotation_compare_difference.png",
    )

    plt.show()


if __name__ == "__main__":
    main()