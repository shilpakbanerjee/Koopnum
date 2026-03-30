"""
Runner for validating CD-kernel reconstruction on a known atomic measure.

This script constructs a finite atomic measure on the unit circle,
computes its exact moments, reconstructs the measure using the
Christoffel–Darboux kernel pipeline, and visualizes the resulting
density proxy. It is intended as a sanity check for the pure-point
case, where the reconstruction should exhibit sharp peaks near the
prescribed atomic locations.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from methods.common.measures.benchmark_measures import AtomicMeasure
from methods.cd_kernel.measure_reconstruction import evaluate_cd_kernel_from_moments


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/measures/atomic")
PLOT_DIR = Path("experiments/cd_kernel/plots/measures/atomic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    order = 60
    grid_size = 2048
    regularization = 1e-8

    # Three atoms on the unit circle with positive weights.
    angles = np.array([0.45, 1.70, 4.10], dtype=float)
    weights = np.array([0.50, 0.30, 0.20], dtype=float)

    mu = AtomicMeasure(angles=angles, weights=weights)
    moments = mu.moments(order=order, normalize=True)

    result = evaluate_cd_kernel_from_moments(
        moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=True,
    )

    print("=== Atomic measure test ===")
    print("Angles:", angles)
    print("Weights:", weights)
    print("Condition number:", result.metadata["toeplitz_condition_number"])
    print("Top peaks:")
    for item in result.top_peaks(k=6, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )

    # Save numerical data
    np.savez(
        OUTPUT_DIR / "atomic_measure_results.npz",
        input_angles=angles,
        input_weights=weights,
        moments=result.moments,
        toeplitz=result.toeplitz,
        angles=result.angles,
        kernel_diag=result.kernel_diag,
        density_proxy=result.density_proxy,
    )

    # Plot density proxy
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(result.angles, result.density_proxy, lw=1.5, label="CD density proxy")
    for j, theta in enumerate(angles):
        ax.axvline(theta, linestyle="--", linewidth=1.0, alpha=0.8, label="true atom" if j == 0 else None)
    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Density proxy")
    ax.set_title("CD-kernel reconstruction of an atomic measure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "atomic_measure_density.png", dpi=160)

    # Plot on unit circle
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    unit_theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax2.plot(np.cos(unit_theta), np.sin(unit_theta), lw=1.0, color="black")
    ax2.scatter(np.cos(angles), np.sin(angles), s=80, marker="o", label="True atoms")
    top_peaks = result.top_peaks(k=6, min_separation=12)
    if top_peaks:
        peak_angles = np.array([item["angle"] for item in top_peaks], dtype=float)
        ax2.scatter(
            np.cos(peak_angles),
            np.sin(peak_angles),
            s=60,
            marker="x",
            label="Detected peaks",
        )
    ax2.set_aspect("equal")
    ax2.set_title("True atoms vs detected peaks")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "atomic_measure_unit_circle.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()