from __future__ import annotations

"""
Finite Koopman approximation on a planar rotation benchmark.

This runner supports two observable modes:

1. OBSERVABLE_MODE = "eigenfunction"
   Observable:
       complex_coordinate(0, 1)
   Interpretation:
       - this is the natural complex eigenfunction for planar rotation
       - the spectral measure should be concentrated at one eigenangle
       - the Krylov space may collapse to very low effective dimension

2. OBSERVABLE_MODE = "rich"
   Observable:
       exp(2π i x)
   Interpretation:
       - this is NOT an eigenfunction of planar rotation
       - its spectral measure contains multiple harmonic atoms
       - the finite Koopman approximation is richer numerically

The purpose of this split is to separate:
- spectral-measure sanity checks
- finite-operator richness checks
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.dynamics.systems import generate_planar_rotation
from experiments.cd_kernel.dynamics.observables import complex_coordinate
from experiments.cd_kernel.dynamics.spectral_measure import spectral_measure_data_from_trajectory
from experiments.cd_kernel.core.kernel import (
    evaluate_cd_kernel_from_moments,
    atomic_mass_proxy_from_kernel,
)
from experiments.cd_kernel.core.koopman import (
    koopman_matrix_from_moments,
    companion_koopman_from_moments,
    spectral_summary,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OBSERVABLE_MODE = "eigenfunction"   # "eigenfunction" or "rich"

N_TRAJ = 3000
ROTATION_ANGLE = 0.35
MOMENT_ORDER = 120
KOOPMAN_ORDER = 30
REGULARIZATION = 1e-6

OUTPUT_DIR = Path("experiments/cd_kernel/outputs/dynamics/koopman_rotation")
PLOT_DIR = Path("experiments/cd_kernel/plots/dynamics/koopman_rotation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Observable selection
# ---------------------------------------------------------------------

def choose_observable(mode: str):
    mode = mode.lower().strip()

    if mode == "eigenfunction":
        observable = complex_coordinate(0, 1)
        description = "complex_coordinate(0, 1)"
        expectation = "single atomic peak near the rotation eigenangle"
        show_expected_single_angle = True
        slug = "eigenfunction"
        return observable, description, expectation, show_expected_single_angle, slug

    if mode == "rich":
        observable = lambda X: np.exp(2j * np.pi * X[:, 0])
        description = "exp(2π i x-coordinate)"
        expectation = "multiple harmonic atoms; richer finite Koopman matrix"
        show_expected_single_angle = False
        slug = "rich"
        return observable, description, expectation, show_expected_single_angle, slug

    raise ValueError("OBSERVABLE_MODE must be 'eigenfunction' or 'rich'")


# ---------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------

def print_spectral_summary(name: str, summary: list[dict]) -> None:
    print(f"\n--- {name} leading eigenvalues ---")
    for item in summary:
        print(
            f"  idx={item['index']:>2d}, "
            f"lambda={item['eigenvalue']}, "
            f"|lambda|={item['modulus']:.6e}, "
            f"arg={item['argument']:.6e}"
        )


def print_cd_peaks(angles: np.ndarray, atomic_proxy: np.ndarray, k: int = 8, min_separation: int = 12) -> None:
    """
    Simple peak extraction directly from the atomic proxy curve.
    """
    y = np.asarray(atomic_proxy, dtype=float)
    peaks = []

    # naive local-max peak picking
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            peaks.append((i, y[i]))

    peaks.sort(key=lambda t: t[1], reverse=True)

    selected = []
    taken = np.zeros(len(y), dtype=bool)

    for idx, val in peaks:
        left = max(0, idx - min_separation)
        right = min(len(y), idx + min_separation + 1)
        if not np.any(taken[left:right]):
            selected.append((idx, val))
            taken[left:right] = True
        if len(selected) >= k:
            break

    print("\n--- CD atomic-proxy peaks ---")
    for idx, val in selected:
        point = np.exp(1j * angles[idx])
        print(
            f"  angle={angles[idx]:.6f}, "
            f"value={val:.6e}, "
            f"point={point}"
        )


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def plot_eigenvalues(
    eigs_main: np.ndarray,
    eigs_companion: np.ndarray,
    true_eig: complex | None,
    show_expected_single_angle: bool,
    save_path,
    title: str,
) -> None:
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

    if show_expected_single_angle and true_eig is not None:
        plt.scatter(
            [np.real(true_eig)],
            [np.imag(true_eig)],
            marker="*",
            s=180,
            label="Expected eigenvalue",
        )

    plt.gca().set_aspect("equal")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)


def plot_singular_values(svals_main: np.ndarray, svals_companion: np.ndarray, save_path, title: str) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.semilogy(svals_main, marker="o", label="Galerkin")
    plt.semilogy(svals_companion, marker="x", label="Companion")
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)


def plot_cd_atomic_proxy(
    angles: np.ndarray,
    atomic_proxy: np.ndarray,
    expected_angle: float | None,
    show_expected_single_angle: bool,
    save_path,
    title: str,
) -> None:
    plt.figure(figsize=(10.0, 4.8))
    plt.plot(angles, atomic_proxy, linewidth=1.5, label="CD atomic proxy")

    if show_expected_single_angle and expected_angle is not None:
        plt.axvline(
            expected_angle,
            linestyle=":",
            linewidth=1.2,
            label="Expected eigenangle",
        )

    plt.xlabel("Angle")
    plt.ylabel("Atomic mass proxy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    observable, observable_description, expectation, show_expected_single_angle, slug = choose_observable(
        OBSERVABLE_MODE
    )

    # -------------------------------------------------------------
    # Generate trajectory and empirical moments
    # -------------------------------------------------------------
    X = generate_planar_rotation(
        n=N_TRAJ,
        theta=ROTATION_ANGLE,
    )

    spec_data = spectral_measure_data_from_trajectory(
        X=X,
        order=MOMENT_ORDER,
        observable=observable,
        center=False,
        normalize=True,
        taper=None,
    )

    moments = spec_data.moments

    # -------------------------------------------------------------
    # CD-kernel side
    # -------------------------------------------------------------
    # Fourier convention fix + modified-kernel path inside core/kernel.py
    moments_for_cd = np.conjugate(moments)
    if np.abs(moments_for_cd[0]) > 1e-14:
        moments_for_cd = moments_for_cd / moments_for_cd[0]

    cd_result = evaluate_cd_kernel_from_moments(
        moments=moments_for_cd,
        order=MOMENT_ORDER,
        grid_size=2048,
        regularization=REGULARIZATION,
        normalize_density=True,
    )

    order_used = int(cd_result.metadata["order_used"])
    atomic_proxy = atomic_mass_proxy_from_kernel(
        cd_result.kernel_diag,
        order=order_used,
    )

    # -------------------------------------------------------------
    # Koopman approximation side
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
    true_eig = np.exp(1j * ROTATION_ANGLE)

    print("\n=== Koopman approximation: planar rotation ===")
    print(f"OBSERVABLE_MODE = {OBSERVABLE_MODE}")
    print(f"Observable = {observable_description}")
    print(f"Expectation = {expectation}")
    print(f"N_TRAJ = {N_TRAJ}")
    print(f"MOMENT_ORDER = {MOMENT_ORDER}")
    print(f"KOOPMAN_ORDER = {KOOPMAN_ORDER}")
    if show_expected_single_angle:
        print(f"Expected eigenvalue = {true_eig}")

    print("\nkernel_diag min/max =", np.min(cd_result.kernel_diag), np.max(cd_result.kernel_diag))
    print("order_used =", order_used)

    print("\n--- Galerkin / solve-based approximation ---")
    for k, v in koopman_main.summary().items():
        print(f"  {k}: {v}")

    print("\n--- Companion approximation ---")
    for k, v in koopman_companion.summary().items():
        print(f"  {k}: {v}")

    print_spectral_summary("Galerkin", spectral_summary(koopman_main, top_k=10))
    print_spectral_summary("Companion", spectral_summary(koopman_companion, top_k=10))
    print_cd_peaks(cd_result.angles, atomic_proxy, k=10, min_separation=12)

    # -------------------------------------------------------------
    # Save arrays
    # -------------------------------------------------------------
    np.savez(
        OUTPUT_DIR / f"koopman_rotation_{slug}_results.npz",
        trajectory=X,
        signal=spec_data.signal,
        moments=moments,
        moments_for_cd=moments_for_cd,
        cd_angles=cd_result.angles,
        cd_kernel_diag=cd_result.kernel_diag,
        cd_atomic_proxy=atomic_proxy,
        main_gram=koopman_main.gram,
        main_shifted_gram=koopman_main.shifted_gram,
        main_koopman=koopman_main.koopman_matrix,
        main_eigenvalues=koopman_main.eigenvalues,
        main_singular_values=koopman_main.singular_values,
        companion_koopman=koopman_companion.koopman_matrix,
        companion_eigenvalues=koopman_companion.eigenvalues,
        companion_singular_values=koopman_companion.singular_values,
        observable_mode=slug,
        rotation_angle=ROTATION_ANGLE,
    )

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------
    plot_eigenvalues(
        eigs_main=koopman_main.eigenvalues,
        eigs_companion=koopman_companion.eigenvalues,
        true_eig=true_eig if show_expected_single_angle else None,
        show_expected_single_angle=show_expected_single_angle,
        save_path=PLOT_DIR / f"koopman_rotation_{slug}_eigs.png",
        title=f"Koopman eigenvalues: planar rotation ({slug})",
    )

    plot_singular_values(
        svals_main=koopman_main.singular_values,
        svals_companion=koopman_companion.singular_values,
        save_path=PLOT_DIR / f"koopman_rotation_{slug}_svals.png",
        title=f"Finite Koopman singular values: planar rotation ({slug})",
    )

    plot_cd_atomic_proxy(
        angles=cd_result.angles,
        atomic_proxy=atomic_proxy,
        expected_angle=ROTATION_ANGLE if show_expected_single_angle else None,
        show_expected_single_angle=show_expected_single_angle,
        save_path=PLOT_DIR / f"koopman_rotation_{slug}_cd_atomic.png",
        title=f"CD-kernel atomic proxy: planar rotation ({slug})",
    )

    plt.show()


if __name__ == "__main__":
    main()