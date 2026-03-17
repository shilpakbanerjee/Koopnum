"""
Runner for applying the CD-kernel pipeline to the Arnold cat map.

This script generates a trajectory of the Arnold cat map on the torus,
evaluates a chosen observable, estimates the moments of the associated
Koopman spectral measure, and reconstructs that measure using the
Christoffel–Darboux kernel method.

The cat map is a standard hyperbolic benchmark whose spectral behavior
for suitable observables is expected to have substantial continuous
components, making it a useful contrast to rigid rotations.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.dynamics.systems import generate_cat_map
from experiments.cd_kernel.dynamics.observables import linear_observable
from experiments.cd_kernel.dynamics.spectral_measure import spectral_measure_data_from_trajectory
from experiments.cd_kernel.measures.reconstruction import evaluate_cd_kernel_from_moments


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/dynamics/catmap")
PLOT_DIR = Path("experiments/cd_kernel/plots/dynamics/catmap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    n = 5000
    order = 80
    grid_size = 2048
    regularization = 1e-6

    X = generate_cat_map(n=n)

    # A simple nontrivial linear observable on the torus coordinates.
    observable = linear_observable([1.0 + 0.0j, 0.5 + 0.0j])

    spec = spectral_measure_data_from_trajectory(
        X,
        order=order,
        observable=observable,
        center=True,
        normalize=True,
    )

    result = evaluate_cd_kernel_from_moments(
        spec.moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=True,
    )

    print("=== Arnold cat map test ===")
    print("n =", n)
    print("Condition number:", result.metadata["toeplitz_condition_number"])
    print("Top peaks:")
    for item in result.top_peaks(k=10, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )

    np.savez(
        OUTPUT_DIR / "catmap_results.npz",
        trajectory=X,
        signal=spec.signal,
        moments=spec.moments,
        toeplitz=result.toeplitz,
        angles=result.angles,
        kernel_diag=result.kernel_diag,
        density_proxy=result.density_proxy,
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(result.angles, result.density_proxy, lw=1.6, label="CD density proxy")
    peaks = result.top_peaks(k=10, min_separation=12)
    if peaks:
        peak_angles = np.array([p["angle"] for p in peaks], dtype=float)
        peak_vals = np.array([p["value"] for p in peaks], dtype=float)
        ax.scatter(peak_angles, peak_vals, marker="x", label="Detected peaks")
    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Density proxy")
    ax.set_title("CD-kernel reconstruction: Arnold cat map")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "catmap_density.png", dpi=160)

    fig2, ax2 = plt.subplots(figsize=(10, 4.8))
    ax2.plot(result.angles, result.kernel_diag, lw=1.4)
    ax2.set_xlabel("Angle on unit circle")
    ax2.set_ylabel("Kernel diagonal")
    ax2.set_title("CD kernel diagonal: Arnold cat map")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "catmap_kernel_diag.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()