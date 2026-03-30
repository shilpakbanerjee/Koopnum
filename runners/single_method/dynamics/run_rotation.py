"""
Runner for applying the CD-kernel pipeline to a planar rotation.

This script generates a trajectory of a rigid planar rotation, evaluates
a chosen observable on the trajectory, estimates the moments of the
associated Koopman spectral measure, and reconstructs that measure using
the Christoffel–Darboux kernel method.

The planar rotation is a pure-point benchmark: the reconstructed spectral
measure should exhibit isolated peaks corresponding to the rotation
frequencies seen by the observable.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from methods.common.systems import generate_planar_rotation
from methods.common.observables import complex_coordinate
from methods.cd_kernel.dynamics.spectral_measure import spectral_measure_data_from_trajectory
from methods.cd_kernel.measure_reconstruction import evaluate_cd_kernel_from_moments


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/dynamics/rotation")
PLOT_DIR = Path("experiments/cd_kernel/plots/dynamics/rotation")
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

    spec = spectral_measure_data_from_trajectory(
        X,
        order=order,
        observable=observable,
        center=False,
        normalize=True,
    )

    result = evaluate_cd_kernel_from_moments(
        spec.moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=True,
    )

    print("=== Planar rotation test ===")
    print("n =", n)
    print("theta =", theta)
    print("Condition number:", result.metadata["toeplitz_condition_number"])
    print("Top peaks:")
    for item in result.top_peaks(k=10, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )

    np.savez(
        OUTPUT_DIR / "rotation_results.npz",
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
    ax.set_title("CD-kernel reconstruction: planar rotation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rotation_density.png", dpi=160)

    fig2, ax2 = plt.subplots(figsize=(10, 4.8))
    ax2.plot(result.angles, result.kernel_diag, lw=1.4)
    ax2.set_xlabel("Angle on unit circle")
    ax2.set_ylabel("Kernel diagonal")
    ax2.set_title("CD kernel diagonal: planar rotation")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "rotation_kernel_diag.png", dpi=160)

    fig3, ax3 = plt.subplots(figsize=(6, 6))
    circle_theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax3.plot(np.cos(circle_theta), np.sin(circle_theta), lw=1.0, color="black")
    if peaks:
        ax3.scatter(np.cos(peak_angles), np.sin(peak_angles), s=60, marker="x", label="Detected peaks")
    ax3.set_aspect("equal")
    ax3.set_title("Detected peaks on the unit circle")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(PLOT_DIR / "rotation_unit_circle.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()