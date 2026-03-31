from __future__ import annotations

"""
CD-kernel runner for an irrational circle rotation.

This runner is meant to model the actual irrational system directly, rather
than replacing it by a periodic rational approximation. The state update is

    x_{n+1} = x_n + alpha mod 1,

with alpha chosen as a floating-point representation of an irrational angle.
In finite-precision arithmetic this is not literally irrational, but for the
trajectory lengths used in our experiments it behaves as the intended
aperiodic rigid rotation.

Two observable modes are provided:

1. OBSERVABLE_MODE = "eigenfunction"
   f(x) = exp(2π i m x)
   Expected spectral measure:
       a single atom at exp(2π i m alpha)

2. OBSERVABLE_MODE = "rich"
   f(x) = c1 exp(2π i x) + c2 exp(2π i 2x) + c3 exp(2π i 3x)
   Expected spectral measure:
       a finite multi-atomic measure supported at the corresponding
       harmonics exp(2π i j alpha)

This runner is primarily for validating the moment/CD-kernel pipeline on a
clean pure-point ergodic benchmark without collapsing the model itself to a
finite periodic system.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from methods.common.systems import generate_torus_translation
from methods.cd_kernel.dynamics.spectral_measure import spectral_measure_data_from_trajectory
from methods.cd_kernel.api import evaluate_cd_kernel_from_moments, atomic_mass_proxy_from_kernel


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ALPHA_NAME = "golden_conjugate"
ALPHA = (np.sqrt(5.0) - 1.0) / 2.0
X0 = 0.123456789
N_TRAJ = 4000
MOMENT_ORDER = 120
GRID_SIZE = 4096
REGULARIZATION = 1e-8
OBSERVABLE_MODE = "rich"   # "eigenfunction" or "rich"

OUTPUT_DIR = Path("outputs/dynamics/irrational_rotation")
PLOT_DIR = Path("outputs/dynamics/irrational_rotation/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Observable selection
# ---------------------------------------------------------------------


def choose_observable(mode: str):
    mode = mode.lower().strip()

    if mode == "eigenfunction":
        harmonic = 1

        def observable(X: np.ndarray) -> np.ndarray:
            x = np.asarray(X, dtype=float).reshape(-1, X.shape[-1] if np.ndim(X) == 2 else 1)[:, 0]
            return np.exp(2j * np.pi * harmonic * x)

        expected_angles = [float((2.0 * np.pi * harmonic * ALPHA) % (2.0 * np.pi))]
        description = f"exp(2π i {harmonic} x)"
        expectation = "single atom at the irrational rotation eigenangle"
        slug = "eigenfunction"
        return observable, description, expectation, expected_angles, slug

    if mode == "rich":
        coeffs = np.array([1.0, 0.35, 0.15], dtype=np.complex128)
        harmonics = np.array([1, 2, 3], dtype=int)

        def observable(X: np.ndarray) -> np.ndarray:
            x = np.asarray(X, dtype=float).reshape(-1, X.shape[-1] if np.ndim(X) == 2 else 1)[:, 0]
            values = np.zeros_like(x, dtype=np.complex128)
            for c, h in zip(coeffs, harmonics):
                values = values + c * np.exp(2j * np.pi * h * x)
            return values

        expected_angles = [float((2.0 * np.pi * h * ALPHA) % (2.0 * np.pi)) for h in harmonics]
        description = "exp(2π i x) + 0.35 exp(2π i 2x) + 0.15 exp(2π i 3x)"
        expectation = "three atomic peaks at the first three harmonics of alpha"
        slug = "rich"
        return observable, description, expectation, expected_angles, slug

    raise ValueError("OBSERVABLE_MODE must be 'eigenfunction' or 'rich'")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def circular_distance(a: float, b: float) -> float:
    return float(np.abs(np.angle(np.exp(1j * (a - b)))))



def print_top_peaks(angles: np.ndarray, atomic_proxy: np.ndarray, k: int = 8) -> list[tuple[float, float]]:
    peaks = []
    y = np.asarray(atomic_proxy, dtype=float)
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            peaks.append((float(angles[i]), float(y[i])))
    peaks.sort(key=lambda item: item[1], reverse=True)
    peaks = peaks[:k]

    print("\n--- leading atomic-proxy peaks ---")
    for angle, value in peaks:
        print(f"  angle={angle:.6f}, value={value:.6e}, point={np.exp(1j * angle)}")
    return peaks



def print_expected_matches(expected_angles: list[float], peaks: list[tuple[float, float]]) -> None:
    if not expected_angles:
        return

    print("\n--- expected harmonic locations ---")
    for idx, expected in enumerate(expected_angles, start=1):
        best_angle, best_value = min(
            peaks,
            key=lambda item: circular_distance(item[0], expected),
        )
        err = circular_distance(best_angle, expected)
        print(
            f"  target[{idx}]={expected:.6f}, nearest_peak={best_angle:.6f}, "
            f"angular_error={err:.6e}, peak_value={best_value:.6e}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    observable, description, expectation, expected_angles, slug = choose_observable(OBSERVABLE_MODE)

    X = generate_torus_translation(
        n=N_TRAJ,
        omega=np.array([ALPHA], dtype=float),
        x0=np.array([X0], dtype=float),
    )

    spec = spectral_measure_data_from_trajectory(
        X=X,
        order=MOMENT_ORDER,
        observable=observable,
        center=False,
        normalize=True,
    )

    cd_result = evaluate_cd_kernel_from_moments(
        spec.moments,
        order=MOMENT_ORDER,
        grid_size=GRID_SIZE,
        regularization=REGULARIZATION,
        normalize_density=True,
    )

    atomic_proxy = atomic_mass_proxy_from_kernel(cd_result.kernel_diag, order=cd_result.metadata["order_used"])
    peaks = print_top_peaks(cd_result.angles, atomic_proxy, k=10)
    print_expected_matches(expected_angles, peaks)

    print("=== Irrational rotation benchmark ===")
    print("alpha_name =", ALPHA_NAME)
    print("alpha =", repr(ALPHA))
    print("x0 =", X0)
    print("n_traj =", N_TRAJ)
    print("moment_order =", MOMENT_ORDER)
    print("observable_mode =", OBSERVABLE_MODE)
    print("observable =", description)
    print("expectation =", expectation)
    print("toeplitz condition number =", cd_result.metadata["toeplitz_condition_number"])

    base = f"irrational_rotation_{slug}"
    np.savez(
        OUTPUT_DIR / f"{base}.npz",
        trajectory=X,
        signal=spec.signal,
        moments=spec.moments,
        angles=cd_result.angles,
        circle_points=cd_result.circle_points,
        kernel_diag=cd_result.kernel_diag,
        density_proxy=cd_result.density_proxy,
        atomic_proxy=atomic_proxy,
        expected_angles=np.array(expected_angles, dtype=float),
        alpha=np.array([ALPHA], dtype=float),
        x0=np.array([X0], dtype=float),
    )

    plt.figure(figsize=(10.5, 4.8))
    plt.plot(cd_result.angles, atomic_proxy, linewidth=1.6, label="CD atomic proxy")
    for idx, angle in enumerate(expected_angles):
        plt.axvline(angle, linestyle=":", linewidth=1.0, label="Expected harmonic" if idx == 0 else None)
    plt.xlabel("Angle on unit circle")
    plt.ylabel("Atomic proxy")
    plt.title(f"Irrational rotation ({OBSERVABLE_MODE})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{base}_atomic_proxy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10.5, 4.8))
    plt.plot(cd_result.angles, cd_result.density_proxy, linewidth=1.5, label="CD density proxy")
    for idx, angle in enumerate(expected_angles):
        plt.axvline(angle, linestyle=":", linewidth=1.0, label="Expected harmonic" if idx == 0 else None)
    plt.xlabel("Angle on unit circle")
    plt.ylabel("Density proxy")
    plt.title(f"Irrational rotation density proxy ({OBSERVABLE_MODE})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{base}_density_proxy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.8, 6.8))
    t = np.linspace(0.0, 2.0 * np.pi, 800)
    plt.plot(np.cos(t), np.sin(t), linewidth=1.0, color="black", label="Unit circle")
    peak_angles = np.array([angle for angle, _ in peaks], dtype=float)
    peak_values = np.array([value for _, value in peaks], dtype=float)
    if peak_angles.size > 0:
        plt.scatter(np.cos(peak_angles), np.sin(peak_angles), s=40 + 25 * peak_values / np.max(peak_values), label="Detected peaks")
    if expected_angles:
        ex = np.array(expected_angles, dtype=float)
        plt.scatter(np.cos(ex), np.sin(ex), marker="x", s=120, label="Expected harmonics")
    plt.gca().set_aspect("equal")
    plt.title(f"Irrational rotation peak locations ({OBSERVABLE_MODE})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{base}_unit_circle.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
