from __future__ import annotations

"""
Finite Koopman approximation on the Arnold cat map benchmark.

This is a continuous-spectrum benchmark:
- the torus Fourier observable exp(2π i x) is natural
- the finite Koopman matrices should not be interpreted as yielding stable
  point spectrum in the same way as the rotation case
- instead, we use this runner to inspect:
    * eigenvalue clouds
    * Gram conditioning
    * singular values
    * relation to the CD-kernel proxy

Outputs
-------
Saved arrays:
    experiments/cd_kernel/outputs/dynamics/koopman_catmap/koopman_catmap_results.npz

Saved plots:
    experiments/cd_kernel/plots/dynamics/koopman_catmap/koopman_catmap_eigs.png
    experiments/cd_kernel/plots/dynamics/koopman_catmap/koopman_catmap_svals.png
    experiments/cd_kernel/plots/dynamics/koopman_catmap/koopman_catmap_cd_density.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.dynamics.systems import generate_cat_map
from experiments.cd_kernel.dynamics.observables import torus_fourier_mode
from experiments.cd_kernel.dynamics.spectral_measure import spectral_measure_data_from_trajectory
from experiments.cd_kernel.core.kernel import evaluate_cd_kernel_from_moments
from experiments.cd_kernel.core.koopman import (
    koopman_matrix_from_moments,
    companion_koopman_from_moments,
    spectral_summary,
)

OUTPUT_DIR = Path("experiments/cd_kernel/outputs/dynamics/koopman_catmap")
PLOT_DIR = Path("experiments/cd_kernel/plots/dynamics/koopman_catmap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

N_TRAJ = 5000
MOMENT_ORDER = 140
KOOPMAN_ORDER = 35
REGULARIZATION = 1e-8


def print_spectral_summary(name: str, summary: list[dict]) -> None:
    print(f"\n--- {name} leading eigenvalues ---")
    for item in summary:
        print(
            f"  idx={item['index']:>2d}, "
            f"lambda={item['eigenvalue']}, "
            f"|lambda|={item['modulus']:.6e}, "
            f"arg={item['argument']:.6e}"
        )


def plot_eigenvalues(eigs_main: np.ndarray, eigs_companion: np.ndarray) -> None:
    plt.figure(figsize=(6.8, 6.8))
    t = np.linspace(0.0, 2.0 * np.pi, 800)
    plt.plot(np.cos(t), np.sin(t), linewidth=1.0, color="black", label="Unit circle")

    plt.scatter(
        np.real(eigs_main),
        np.imag(eigs_main),
        marker="o",
        s=28,
        alpha=0.8,
        label="Galerkin eigenvalues",
    )

    plt.scatter(
        np.real(eigs_companion),
        np.imag(eigs_companion),
        marker="x",
        s=36,
        alpha=0.8,
        label="Companion eigenvalues",
    )

    plt.gca().set_aspect("equal")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Koopman eigenvalue cloud: cat map")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "koopman_catmap_eigs.png", dpi=160)


def plot_singular_values(svals_main: np.ndarray, svals_companion: np.ndarray) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.semilogy(svals_main, marker="o", label="Galerkin")
    plt.semilogy(svals_companion, marker="x", label="Companion")
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Finite Koopman singular values: cat map")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "koopman_catmap_svals.png", dpi=160)


def plot_cd_density(cd_result) -> None:
    plt.figure(figsize=(10.0, 4.8))
    plt.plot(cd_result.angles, cd_result.density_proxy, linewidth=1.5, label="CD density proxy")
    plt.xlabel("Angle")
    plt.ylabel("Density proxy")
    plt.title("CD-kernel proxy: cat map")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "koopman_catmap_cd_density.png", dpi=160)


def main() -> None:
    # -------------------------------------------------------------
    # Generate data and empirical moments
    # -------------------------------------------------------------
    X = generate_cat_map(n=N_TRAJ)
    observable = torus_fourier_mode(1, 0)

    spec_data = spectral_measure_data_from_trajectory(
        X=X,
        order=MOMENT_ORDER,
        observable=observable,
        center=True,
        normalize=True,
        taper=None,
    )

    moments = spec_data.moments

    # --- Fix Fourier convention (CRITICAL) ---
    moments_for_cd = np.conjugate(moments)

    # -------------------------------------------------------------
    # Spectral measure proxy for comparison
    # -------------------------------------------------------------
    cd_result = evaluate_cd_kernel_from_moments(
        moments=moments,
        order=MOMENT_ORDER,
        grid_size=2048,
        regularization=REGULARIZATION,
        normalize_density=True,
    )

    # -------------------------------------------------------------
    # Koopman approximations
    # -------------------------------------------------------------
    koopman_main = koopman_matrix_from_moments(
        moments=moments,
        order=KOOPMAN_ORDER,
        regularization=REGULARIZATION,
        solve_method="solve",
    )

    koopman_companion = companion_koopman_from_moments(
        moments=moments,
        order=KOOPMAN_ORDER,
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------
    print("\n=== Koopman approximation: cat map ===")
    print(f"N_TRAJ = {N_TRAJ}")
    print(f"MOMENT_ORDER = {MOMENT_ORDER}")
    print(f"KOOPMAN_ORDER = {KOOPMAN_ORDER}")
    print("Observable = torus_fourier_mode(1, 0)")

    print("\n--- Galerkin / solve-based approximation ---")
    for k, v in koopman_main.summary().items():
        print(f"  {k}: {v}")

    print("\n--- Companion approximation ---")
    for k, v in koopman_companion.summary().items():
        print(f"  {k}: {v}")

    print_spectral_summary("Galerkin", spectral_summary(koopman_main, top_k=12))
    print_spectral_summary("Companion", spectral_summary(koopman_companion, top_k=12))

    print("\n--- CD peaks ---")
    for item in cd_result.top_peaks(k=10, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )

    # -------------------------------------------------------------
    # Save arrays
    # -------------------------------------------------------------
    np.savez(
        OUTPUT_DIR / "koopman_catmap_results.npz",
        trajectory=X,
        signal=spec_data.signal,
        moments=moments,
        cd_angles=cd_result.angles,
        cd_density=cd_result.density_proxy,
        main_gram=koopman_main.gram,
        main_shifted_gram=koopman_main.shifted_gram,
        main_koopman=koopman_main.koopman_matrix,
        main_eigenvalues=koopman_main.eigenvalues,
        main_singular_values=koopman_main.singular_values,
        companion_koopman=koopman_companion.koopman_matrix,
        companion_eigenvalues=koopman_companion.eigenvalues,
        companion_singular_values=koopman_companion.singular_values,
    )

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------
    plot_eigenvalues(
        eigs_main=koopman_main.eigenvalues,
        eigs_companion=koopman_companion.eigenvalues,
    )

    plot_singular_values(
        svals_main=koopman_main.singular_values,
        svals_companion=koopman_companion.singular_values,
    )

    plot_cd_density(cd_result)

    plt.show()


if __name__ == "__main__":
    main()