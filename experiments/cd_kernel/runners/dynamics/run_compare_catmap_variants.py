"""
Compare baseline and tapered CD-kernel variants on the Arnold cat map.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.core.diagnostics import (
    summarize_result,
    compare_results,
    weak_convergence_summary,
)
from experiments.cd_kernel.dynamics.systems import generate_cat_map
from experiments.cd_kernel.dynamics.observables import torus_fourier_mode
from experiments.cd_kernel.dynamics.spectral_measure import (
    reconstruct_spectral_measure_from_system,
)
from experiments.cd_kernel.variants.cd_kernel_v002_tapered import (
    fit_cd_kernel_tapered_from_moments,
)
from experiments.cd_kernel.runners.common_plotting import (
    save_density_comparison_plot,
    save_density_comparison_log_plot,
    save_density_comparison_normalized_plot,
    save_kernel_comparison_plot,
    save_peak_overlay_plot,
    save_difference_plot,
)


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/dynamics/catmap_compare")
PLOT_DIR = Path("experiments/cd_kernel/plots/dynamics/catmap_compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    n = 5000
    order = 80
    grid_size = 2048
    regularization = 1e-6

    observable = torus_fourier_mode(1, 0)

    X, baseline_spec, baseline = reconstruct_spectral_measure_from_system(
        system_fn=generate_cat_map,
        system_kwargs={"n": n},
        order=order,
        observable=observable,
        center=True,
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

    print("=== Arnold cat map: baseline vs tapered ===")
    print("n =", n)
    print("observable = torus_fourier_mode(1, 0)")

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
        OUTPUT_DIR / "catmap_compare_results.npz",
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
        title="Arnold cat map: baseline vs tapered CD kernel",
        save_path=PLOT_DIR / "catmap_compare_density_linear.png",
        styles=styles,
    )

    save_density_comparison_log_plot(
        results=[baseline, tapered],
        labels=["Baseline", "Tapered (Fejér)"],
        title="Arnold cat map: density comparison (log scale)",
        save_path=PLOT_DIR / "catmap_compare_density_log.png",
        styles=styles,
    )

    save_density_comparison_normalized_plot(
        results=[baseline, tapered],
        labels=["Baseline / max", "Tapered / max"],
        title="Arnold cat map: max-normalized comparison",
        save_path=PLOT_DIR / "catmap_compare_density_normalized.png",
        styles=styles,
    )

    save_kernel_comparison_plot(
        results=[baseline, tapered],
        labels=["Baseline kernel", "Tapered kernel"],
        title="Arnold cat map: kernel diagonal comparison",
        save_path=PLOT_DIR / "catmap_compare_kernel.png",
        styles=styles,
    )

    save_peak_overlay_plot(
        base_result=tapered,
        overlay_results=[baseline, tapered],
        overlay_labels=["Baseline peaks", "Tapered peaks"],
        title="Arnold cat map: detected peaks",
        save_path=PLOT_DIR / "catmap_compare_peaks.png",
        base_label="Tapered density",
    )

    save_difference_plot(
        result_a=baseline,
        result_b=tapered,
        title="Arnold cat map: |baseline density - tapered density|",
        save_path=PLOT_DIR / "catmap_compare_difference.png",
    )

    plt.show()


if __name__ == "__main__":
    main()


