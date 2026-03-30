"""
Runner for validating CD-kernel reconstruction on the uniform measure.

This script tests the reconstruction pipeline on the normalized Lebesgue
measure on the unit circle. Since the underlying measure is absolutely
continuous with constant density, the reconstructed density proxy should
be comparatively flat, without isolated atomic spikes.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from methods.common.measures.benchmark_measures import (
    AbsolutelyContinuousMeasure,
    uniform_density,
)
from methods.cd_kernel.measure_reconstruction import evaluate_cd_kernel_from_moments


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/measures/uniform")
PLOT_DIR = Path("experiments/cd_kernel/plots/measures/uniform")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    order = 60
    grid_size = 2048
    moment_grid_size = 4096
    regularization = 1e-8

    mu = AbsolutelyContinuousMeasure(uniform_density)
    moments = mu.moments(order=order, grid_size=moment_grid_size, normalize=True)

    result = evaluate_cd_kernel_from_moments(
        moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=True,
    )

    print("=== Uniform measure test ===")
    print("Condition number:", result.metadata["toeplitz_condition_number"])
    print("Top peaks:")
    for item in result.top_peaks(k=6, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )

    np.savez(
        OUTPUT_DIR / "uniform_measure_results.npz",
        moments=result.moments,
        toeplitz=result.toeplitz,
        angles=result.angles,
        kernel_diag=result.kernel_diag,
        density_proxy=result.density_proxy,
    )

    # Theoretical density for comparison
    true_density = np.ones_like(result.angles, dtype=float) / (2.0 * np.pi)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(result.angles, result.density_proxy, lw=1.5, label="CD density proxy")
    ax.plot(result.angles, true_density, linestyle="--", linewidth=1.2, label="True uniform density")
    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Density")
    ax.set_title("CD-kernel reconstruction of the uniform measure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "uniform_measure_density.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()