from __future__ import annotations

"""
CD-kernel runner for periodic approximants to an irrational circle rotation.

This script studies the approximation scheme

    alpha_n = p_n / q_n -> alpha,

where alpha is the golden-ratio conjugate and alpha_n are its continued-
fraction convergents. Each approximant defines a genuinely periodic circle
rotation, so the resulting spectral measure is finite atomic on q_n-th roots
of unity. The point of this runner is not to replace the irrational model,
but to examine how the CD-kernel output changes across periodic stages.

Two observable modes are supported, matching the irrational runner:

1. OBSERVABLE_MODE = "eigenfunction"
   f(x) = exp(2π i x)
   Expected spectral measure for alpha_n:
       a single atom at exp(2π i alpha_n)

2. OBSERVABLE_MODE = "rich"
   f(x) = exp(2π i x) + 0.35 exp(2π i 2x) + 0.15 exp(2π i 3x)
   Expected spectral measure for alpha_n:
       atoms at the first few harmonics of alpha_n, with possible aliasing
       if harmonics collide modulo q_n.
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

TARGET_NAME = "golden_conjugate"
TARGET_ALPHA = (np.sqrt(5.0) - 1.0) / 2.0
X0 = 0.123456789
MOMENT_ORDER = 120
GRID_SIZE = 4096
REGULARIZATION = 1e-8
OBSERVABLE_MODE = "rich"   # "eigenfunction" or "rich"
MAX_CONVERGENTS = 6

OUTPUT_DIR = Path("outputs/dynamics/rotation_rational_approximants")
PLOT_DIR = Path("outputs/dynamics/rotation_rational_approximants/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Continued-fraction approximants for the golden conjugate
# ---------------------------------------------------------------------


def fibonacci_convergents(count: int) -> list[tuple[int, int]]:
    if count < 1:
        raise ValueError("count must be positive")

    fib = [1, 1]
    while len(fib) < count + 2:
        fib.append(fib[-1] + fib[-2])

    pairs: list[tuple[int, int]] = []
    for n in range(2, count + 2):
        pairs.append((fib[n - 2], fib[n - 1]))
    return pairs


# ---------------------------------------------------------------------
# Observable selection
# ---------------------------------------------------------------------


def choose_observable(mode: str):
    mode = mode.lower().strip()

    if mode == "eigenfunction":
        harmonics = np.array([1], dtype=int)
        coeffs = np.array([1.0], dtype=np.complex128)
        description = "exp(2π i x)"
        expectation = "single atom at the rotation eigenangle for each approximant"
        slug = "eigenfunction"
    elif mode == "rich":
        harmonics = np.array([1, 2, 3], dtype=int)
        coeffs = np.array([1.0, 0.35, 0.15], dtype=np.complex128)
        description = "exp(2π i x) + 0.35 exp(2π i 2x) + 0.15 exp(2π i 3x)"
        expectation = "few-atom spectral measure; possible aliasing at low denominators"
        slug = "rich"
    else:
        raise ValueError("OBSERVABLE_MODE must be 'eigenfunction' or 'rich'")

    def observable(X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float).reshape(-1, X.shape[-1] if np.ndim(X) == 2 else 1)[:, 0]
        values = np.zeros_like(x, dtype=np.complex128)
        for c, h in zip(coeffs, harmonics):
            values = values + c * np.exp(2j * np.pi * h * x)
        return values

    return observable, description, expectation, harmonics, coeffs, slug


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def circular_distance(a: float, b: float) -> float:
    return float(np.abs(np.angle(np.exp(1j * (a - b)))))



def expected_angles_for_alpha(alpha: float, harmonics: np.ndarray) -> np.ndarray:
    return np.array([(2.0 * np.pi * h * alpha) % (2.0 * np.pi) for h in harmonics], dtype=float)



def trajectory_length_for_denominator(q: int, min_cycles: int = 24) -> int:
    return max(3000, min_cycles * q)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    observable, description, expectation, harmonics, coeffs, slug = choose_observable(OBSERVABLE_MODE)
    convergents = fibonacci_convergents(MAX_CONVERGENTS)

    summary_rows: list[tuple[int, int, float, float, float, float, int]] = []
    density_curves = []
    atomic_curves = []

    print("=== Rational approximants to irrational rotation ===")
    print("target_name =", TARGET_NAME)
    print("target_alpha =", repr(TARGET_ALPHA))
    print("observable_mode =", OBSERVABLE_MODE)
    print("observable =", description)
    print("expectation =", expectation)

    for idx, (p, q) in enumerate(convergents, start=1):
        alpha_n = p / q
        n_traj = trajectory_length_for_denominator(q)
        expected_angles = expected_angles_for_alpha(alpha_n, harmonics)

        X = generate_torus_translation(
            n=n_traj,
            omega=np.array([alpha_n], dtype=float),
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
        peaks = cd_result.top_peaks(k=8, min_separation=16)
        peak_angles = np.array([item["angle"] for item in peaks], dtype=float)
        peak_values = np.array([item["value"] for item in peaks], dtype=float)

        density_curves.append((f"{p}/{q}", cd_result.angles.copy(), cd_result.density_proxy.copy()))
        atomic_curves.append((f"{p}/{q}", cd_result.angles.copy(), atomic_proxy.copy()))

        worst_match = 0.0
        if peak_angles.size > 0:
            for target in expected_angles:
                nearest = float(np.min([circular_distance(target, angle) for angle in peak_angles]))
                worst_match = max(worst_match, nearest)

        alpha_error = abs(alpha_n - TARGET_ALPHA)
        summary_rows.append(
            (
                p,
                q,
                alpha_n,
                alpha_error,
                float(cd_result.metadata["toeplitz_condition_number"]),
                worst_match,
                n_traj,
            )
        )

        print(f"\n--- convergent {idx}: {p}/{q} ---")
        print("alpha_n =", repr(alpha_n))
        print("|alpha_n - alpha| =", alpha_error)
        print("trajectory_length =", n_traj)
        print("toeplitz condition number =", cd_result.metadata["toeplitz_condition_number"])
        print("expected_angles =", [float(a) for a in expected_angles])
        print("top peaks:")
        for item in peaks:
            print(
                f"  angle={item['angle']:.6f}, value={item['value']:.6e}, point={item['point']}"
            )

        out_name = f"rotation_approx_{slug}_{idx:02d}_{p}_over_{q}.npz"
        np.savez(
            OUTPUT_DIR / out_name,
            p=np.array([p], dtype=int),
            q=np.array([q], dtype=int),
            alpha_n=np.array([alpha_n], dtype=float),
            target_alpha=np.array([TARGET_ALPHA], dtype=float),
            trajectory=X,
            signal=spec.signal,
            moments=spec.moments,
            angles=cd_result.angles,
            kernel_diag=cd_result.kernel_diag,
            density_proxy=cd_result.density_proxy,
            atomic_proxy=atomic_proxy,
            expected_angles=expected_angles,
            harmonics=harmonics,
            coefficients=coeffs,
        )

    summary_path = OUTPUT_DIR / f"rotation_approximants_summary_{slug}.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("p,q,alpha_n,alpha_error,toeplitz_condition_number,worst_peak_miss,trajectory_length\n")
        for row in summary_rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.18e},{row[3]:.18e},{row[4]:.18e},{row[5]:.18e},{row[6]}\n"
            )

    plt.figure(figsize=(10.5, 5.2))
    for label, angles, atomic_proxy in atomic_curves:
        plt.plot(angles, atomic_proxy, linewidth=1.2, label=label)
    plt.xlabel("Angle on unit circle")
    plt.ylabel("Atomic proxy")
    plt.title(f"Periodic approximants of irrational rotation ({OBSERVABLE_MODE})")
    plt.grid(True, alpha=0.3)
    plt.legend(title="p/q", ncol=2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"rotation_approximants_atomic_proxy_{slug}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10.5, 5.2))
    for label, angles, density_proxy in density_curves:
        plt.plot(angles, density_proxy, linewidth=1.2, label=label)
    plt.xlabel("Angle on unit circle")
    plt.ylabel("Density proxy")
    plt.title(f"Density proxy across periodic approximants ({OBSERVABLE_MODE})")
    plt.grid(True, alpha=0.3)
    plt.legend(title="p/q", ncol=2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"rotation_approximants_density_proxy_{slug}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8.2, 4.8))
    denominators = np.array([row[1] for row in summary_rows], dtype=float)
    alpha_errors = np.array([row[3] for row in summary_rows], dtype=float)
    peak_misses = np.array([row[5] for row in summary_rows], dtype=float)
    plt.loglog(denominators, alpha_errors, marker="o", label=r"$|p/q - \alpha|$")
    plt.loglog(denominators, np.maximum(peak_misses, 1e-16), marker="x", label="Worst peak miss")
    plt.xlabel("Denominator q")
    plt.ylabel("Error scale")
    plt.title("Approximation quality across convergents")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"rotation_approximants_errors_{slug}.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
