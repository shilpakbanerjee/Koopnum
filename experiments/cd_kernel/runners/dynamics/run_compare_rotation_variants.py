"""
Compare baseline and tapered CD-kernel variants on planar rotation.

This runner applies both:
- v001 baseline CD kernel
- v002 tapered CD kernel

to the same planar rotation trajectory and overlays the reconstructed
density proxies and detected peaks. It is intended as the first direct
algorithm-comparison experiment.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.dynamics.systems import generate_planar_rotation
from experiments.cd_kernel.dynamics.observables import complex_coordinate
from experiments.cd_kernel.variants.cd_kernel_v001_baseline import fit_cd_kernel_baseline
from experiments.cd_kernel.variants.cd_kernel_v002_tapered import fit_cd_kernel_tapered
from experiments.cd_kernel.runners.common_plotting import (
    save_density_comparison_plot,
    save_kernel_comparison_plot,
    save_peak_overlay_plot,
    save_unit_circle_peaks_plot,
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

    X = generate_planar_rotation(n=n, theta=theta)
    observable = complex_coordinate(0, 1)

    baseline = fit_cd_kernel_baseline(
        X,
        order=order,
        observable=observable,
        grid_size=grid_size,
        regularization=regularization,
        center=False,
        normalize_moments=True,
        normalize_density=True,
    )

    tapered = fit_cd_kernel_tapered(
        X,
        order=order,
        observable=observable,
        grid_size=grid_size,
        regularization=regularization,
        taper="fejer",
        center=False,
        normalize_moments=True,
        normalize_density=True,
    )

    expected_angle = (2.0 * np.pi - theta) % (2.0 * np.pi)

    print("=== Rotation: baseline vs tapered ===")
    print("theta =", theta)
    print("Expected dominant angle (current convention) =", expected_angle)

    print("Baseline top peaks:")
    for item in baseline.top_peaks(k=8, min_separation=12):
        print(f"  angle={item['angle']:.6f}, value={item['value']:.6e}")

    print("Tapered top peaks:")
    for item in tapered.top_peaks(k=8, min_separation=12):
        print(f"  angle={item['angle']:.6f}, value={item['value']:.6e}")

    np.savez(
        OUTPUT_DIR / "rotation_compare_results.npz",
        trajectory=X,
        baseline_angles=baseline.angles,
        baseline_density=baseline.density_proxy,
        baseline_kernel=baseline.kernel_diag,
        tapered_angles=tapered.angles,
        tapered_density=tapered.density_proxy,
        tapered_kernel=tapered.kernel_diag,
        expected_angle=expected_angle,
    )

    save_density_comparison_plot(
        results=[baseline, tapered],
        labels=["Baseline", "Tapered (Fejér)"],
        title="Planar rotation: baseline vs tapered CD kernel",
        save_path=PLOT_DIR / "rotation_compare_density.png",
        show_peaks=True,
    )

    save_kernel_comparison_plot(
        results=[baseline, tapered],
        labels=["Baseline kernel", "Tapered kernel"],
        title="Kernel diagonal: baseline vs tapered",
        save_path=PLOT_DIR / "rotation_compare_kernel.png",
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

    plt.show()


if __name__ == "__main__":
    main()