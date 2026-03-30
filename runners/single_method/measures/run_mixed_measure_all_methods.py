from __future__ import annotations

"""
Mixed-measure benchmark for moment-based spectral reconstruction methods.

Purpose
-------
This runner compares three reconstruction methods on a benchmark measure
with both absolutely continuous and atomic components.

The benchmark is useful because it cleanly separates the roles of the
three reconstruction branches currently implemented in the project:

1. CD-kernel reconstruction:
   A density-like spectral detection tool that highlights atoms strongly.

2. Cesàro / Fejér reconstruction:
   An absolutely continuous weak approximation built from Fourier moments.

3. Quadrature reconstruction:
   A purely atomic weak approximation built by nonnegative moment matching.

Why this benchmark matters
--------------------------
The mixed benchmark is a clean reference point before moving to more subtle
settings such as singular continuous measures or dynamical spectral measures.
It helps answer:

- Does the CD proxy detect atoms?
- Does Cesàro recover the broad AC structure?
- Does quadrature recover weak CDF information and atom locations?

Outputs
-------
Plots:
    - mixed_all_methods_density.png
    - mixed_all_methods_cdf.png
    - mixed_all_methods_unit_circle_atoms.png

Data:
    - mixed_all_methods_results.npz

Printed diagnostics:
    - CD peak information
    - Cesàro and quadrature CDF errors vs the true benchmark
    - recovered quadrature atoms and residuals
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from methods.common.measures.benchmark_measures import (
    AtomicMeasure,
    AbsolutelyContinuousMeasure,
    wrapped_gaussian_density,
)
from methods.common.moments import normalize_moments
from methods.common.measures.quadrature import (
    significant_atoms,
)
from methods.cd_kernel.measure_api import (
    run_all_measure_methods_from_moments,
)

Array = np.ndarray

OUTPUT_DIR = Path("experiments/cd_kernel/outputs/measures/mixed_all_methods")
PLOT_DIR = Path("experiments/cd_kernel/plots/measures/mixed_all_methods")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ORDER = 80
GRID_SIZE = 2048
MOMENT_GRID_SIZE = 4096
REGULARIZATION = 1e-8

ATOMIC_ANGLES = np.array([0.55, 2.15, 5.00], dtype=float)
ATOMIC_WEIGHTS = np.array([0.18, 0.12, 0.10], dtype=float)
ATOMIC_WEIGHTS = ATOMIC_WEIGHTS / np.sum(ATOMIC_WEIGHTS)

ATOMIC_MASS = 0.40
AC_MASS = 0.60

QUADRATURE_NODE_MULTIPLIER = 6
QUADRATURE_MASS_CONSTRAINT_WEIGHT = 5.0


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def circle_integral(values: Array, angles: Array) -> float:
    values = np.asarray(values, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, angles))
    return float(np.trapz(values, angles))


def cumulative_from_density(density: Array, angles: Array) -> Array:
    density = np.asarray(density, dtype=float)
    angles = np.asarray(angles, dtype=float)

    if density.shape != angles.shape:
        raise ValueError("density and angles must have the same shape")
    if len(angles) < 2:
        return np.zeros_like(angles)

    dtheta = float(angles[1] - angles[0])
    return np.cumsum(density) * dtheta


def normalize_density_values(values: Array, angles: Array, target_mass: float = 1.0) -> Array:
    values = np.asarray(values, dtype=float)
    mass = circle_integral(values, angles)
    if mass <= 0:
        raise ValueError("density integral must be positive")
    return values * (target_mass / mass)


def mixed_ac_density(
    weight1: float = 0.55,
    center1: float = 1.2,
    sigma1: float = 0.24,
    weight2: float = 0.45,
    center2: float = 4.5,
    sigma2: float = 0.38,
):
    """
    Smooth AC density on the unit circle, normalized to integrate to 1.
    """
    if weight1 < 0 or weight2 < 0:
        raise ValueError("weights must be nonnegative")
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("sigmas must be positive")

    g1 = wrapped_gaussian_density(center=center1, sigma=sigma1)
    g2 = wrapped_gaussian_density(center=center2, sigma=sigma2)

    def rho(angles: Array) -> Array:
        raw = weight1 * g1(angles) + weight2 * g2(angles)
        raw = np.asarray(raw, dtype=float)
        return normalize_density_values(raw, angles, target_mass=1.0)

    return rho


def exact_mixed_moments(
    order: int,
    atomic_angles: Array,
    atomic_weights: Array,
    ac_density_fn,
    atomic_mass: float,
    ac_mass: float,
    ac_grid_size: int = 4096,
) -> Array:
    """
    Construct exact normalized moments of a mixed measure:
        mu = ac_mass * mu_ac + atomic_mass * mu_at
    """
    mu_at = AtomicMeasure(
        angles=np.asarray(atomic_angles, dtype=float),
        weights=np.asarray(atomic_weights, dtype=float),
    )
    mu_ac = AbsolutelyContinuousMeasure(ac_density_fn)

    m_at = mu_at.moments(order=order, normalize=True)
    m_ac = mu_ac.moments(order=order, grid_size=ac_grid_size, normalize=True)

    moments = atomic_mass * m_at + ac_mass * m_ac
    return normalize_moments(moments)


def exact_true_ac_density(
    angles: Array,
    ac_density_fn,
    ac_mass: float,
) -> Array:
    """
    Evaluate the true AC density component on a grid.
    """
    rho = np.asarray(ac_density_fn(angles), dtype=float)
    rho = normalize_density_values(rho, angles, target_mass=1.0)
    return ac_mass * rho


def exact_true_cdf(
    angles: Array,
    atomic_angles: Array,
    atomic_weights: Array,
    ac_density_fn,
    atomic_mass: float,
    ac_mass: float,
) -> Array:
    """
    Evaluate the true benchmark CDF on a grid.
    """
    rho_ac = exact_true_ac_density(
        angles,
        ac_density_fn=ac_density_fn,
        ac_mass=ac_mass,
    )
    cdf_ac = cumulative_from_density(rho_ac, angles)

    atomic_angles = np.asarray(atomic_angles, dtype=float)
    atomic_weights = np.asarray(atomic_weights, dtype=float)

    atom_jumps = np.zeros_like(angles, dtype=float)
    scaled_weights = atomic_mass * atomic_weights
    for theta, w in zip(atomic_angles, scaled_weights):
        atom_jumps += w * (angles >= theta)

    return cdf_ac + atom_jumps


def step_cdf_on_grid(step_x: Array, step_y: Array, grid: Array) -> Array:
    """
    Evaluate a right-continuous step CDF on a target grid.
    """
    step_x = np.asarray(step_x, dtype=float)
    step_y = np.asarray(step_y, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if len(step_x) == 0:
        return np.zeros_like(grid)

    idx = np.searchsorted(step_x, grid, side="right") - 1
    out = np.zeros_like(grid, dtype=float)
    valid = idx >= 0
    out[valid] = step_y[idx[valid]]
    return out


def l1_error(y_true: Array, y_pred: Array, x: Array) -> float:
    e = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return circle_integral(e, x)


def print_cd_peaks(cd_result, k: int = 8, min_separation: int = 12) -> None:
    print("\n--- CD kernel peaks ---")
    for item in cd_result.top_peaks(k=k, min_separation=min_separation):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )


def print_quadrature_atoms(quad_atoms: list[dict], max_items: int = 20) -> None:
    print("\n--- Significant quadrature atoms (weight > 1e-3) ---")
    for atom in quad_atoms[:max_items]:
        print(
            f"  angle={atom['angle']:.6f}, "
            f"weight={atom['weight']:.6e}, "
            f"point={atom['point']}"
        )


def plot_density(
    angles: Array,
    cd_density: Array,
    cesaro_density: Array,
    true_ac_density: Array,
    atomic_angles: Array,
) -> None:
    """
    Compare density-like outputs:
    - CD density proxy
    - Cesàro density
    - true AC density component
    """
    plt.figure(figsize=(10.5, 5.0))
    plt.plot(angles, cd_density, linewidth=1.6, label="CD density proxy")
    plt.plot(angles, cesaro_density, linewidth=1.6, label="Cesàro density")
    plt.plot(
        angles,
        true_ac_density,
        linestyle="--",
        linewidth=1.4,
        label="True AC density component",
    )
    for j, theta in enumerate(atomic_angles):
        plt.axvline(
            theta,
            linestyle=":",
            linewidth=1.0,
            alpha=0.9,
            label="True atom location" if j == 0 else None,
        )

    plt.xlabel("Angle on unit circle")
    plt.ylabel("Density / density proxy")
    plt.title("Mixed benchmark: CD proxy vs Cesàro density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mixed_all_methods_density.png", dpi=160)


def plot_cdf(
    angles: Array,
    true_cdf: Array,
    cesaro_cdf: Array,
    quad_cdf: Array,
) -> None:
    """
    Compare CDF-level weak reconstructions.
    """
    plt.figure(figsize=(10.5, 5.0))
    plt.plot(angles, true_cdf, linewidth=1.8, label="True CDF")
    plt.plot(angles, cesaro_cdf, linewidth=1.6, label="Cesàro CDF")
    plt.step(
        angles,
        quad_cdf,
        where="post",
        linewidth=1.3,
        label="Quadrature CDF",
    )

    plt.xlabel("Angle on unit circle")
    plt.ylabel("CDF")
    plt.title("Mixed benchmark: weak reconstructions at CDF level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mixed_all_methods_cdf.png", dpi=160)


def plot_unit_circle_atoms(
    true_atom_angles: Array,
    true_scaled_weights: Array,
    recovered_angles: Array,
    recovered_weights: Array,
) -> None:
    """
    Plot true atoms and recovered quadrature atoms on the unit circle.
    """
    plt.figure(figsize=(6.5, 6.5))
    theta_circle = np.linspace(0.0, 2.0 * np.pi, 800)
    plt.plot(np.cos(theta_circle), np.sin(theta_circle), linewidth=1.0, color="black")

    true_sizes = 800.0 * true_scaled_weights / np.max(true_scaled_weights)
    plt.scatter(
        np.cos(true_atom_angles),
        np.sin(true_atom_angles),
        s=true_sizes,
        marker="o",
        label="True atoms",
    )

    if len(recovered_angles) > 0:
        recovered_sizes = 500.0 * recovered_weights / np.max(recovered_weights)
        plt.scatter(
            np.cos(recovered_angles),
            np.sin(recovered_angles),
            s=recovered_sizes,
            marker="x",
            label="Quadrature atoms",
        )

    plt.gca().set_aspect("equal")
    plt.title("True atoms vs quadrature-recovered atoms")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mixed_all_methods_unit_circle_atoms.png", dpi=160)


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def main() -> None:
    # -------------------------------------------------------------
    # Build benchmark measure and exact moments
    # -------------------------------------------------------------
    ac_density_fn = mixed_ac_density(
        weight1=0.58,
        center1=1.35,
        sigma1=0.22,
        weight2=0.42,
        center2=4.55,
        sigma2=0.36,
    )

    moments = exact_mixed_moments(
        order=ORDER,
        atomic_angles=ATOMIC_ANGLES,
        atomic_weights=ATOMIC_WEIGHTS,
        ac_density_fn=ac_density_fn,
        atomic_mass=ATOMIC_MASS,
        ac_mass=AC_MASS,
        ac_grid_size=MOMENT_GRID_SIZE,
    )

    # -------------------------------------------------------------
    # Three reconstructions from the same moment data
    # -------------------------------------------------------------
    results = run_all_measure_methods_from_moments(
        moments,
        order=ORDER,
        grid_size=GRID_SIZE,
        cd_normalize_density=True,
        cesaro_clip_negative=True,
        cesaro_normalize_mass=True,
        quadrature_nodes=QUADRATURE_NODE_MULTIPLIER * (ORDER + 1),
        quadrature_mass_constraint_weight=QUADRATURE_MASS_CONSTRAINT_WEIGHT,
        quadrature_normalize_mass=True,
    )

    cd_result = results["cd"]
    cesaro_result = results["cesaro"]
    quad_result = results["quadrature"]

    # -------------------------------------------------------------
    # Exact benchmark references on the same grid
    # -------------------------------------------------------------
    angles = cd_result.angles

    true_ac_density = exact_true_ac_density(
        angles,
        ac_density_fn=ac_density_fn,
        ac_mass=AC_MASS,
    )

    true_cdf = exact_true_cdf(
        angles,
        atomic_angles=ATOMIC_ANGLES,
        atomic_weights=ATOMIC_WEIGHTS,
        ac_density_fn=ac_density_fn,
        atomic_mass=ATOMIC_MASS,
        ac_mass=AC_MASS,
    )

    cesaro_density = np.asarray(cesaro_result.density, dtype=float)
    cesaro_cdf = np.asarray(cesaro_result.cdf, dtype=float)

    quad_cdf_on_angles = step_cdf_on_grid(
        quad_result.cdf_grid,
        quad_result.cdf_values,
        angles,
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------
    cesaro_cdf_l1 = l1_error(true_cdf, cesaro_cdf, angles)
    quad_cdf_l1 = l1_error(true_cdf, quad_cdf_on_angles, angles)

    quad_atoms = significant_atoms(quad_result, tol=1e-3)

    print("\n=== Mixed measure all-method comparison ===")
    print(f"order = {ORDER}")
    print(f"grid_size = {GRID_SIZE}")
    print(f"atomic_mass = {ATOMIC_MASS:.3f}, ac_mass = {AC_MASS:.3f}")
    print("true atomic angles =", ATOMIC_ANGLES)
    print("true atomic weights (scaled) =", ATOMIC_MASS * ATOMIC_WEIGHTS)

    print("\n--- CD kernel ---")
    print("Toeplitz condition number:", cd_result.metadata["toeplitz_condition_number"])

    print("\n--- Cesàro ---")
    print("Approximate final CDF value:", float(cesaro_cdf[-1]))
    print("L1 CDF error vs true benchmark:", cesaro_cdf_l1)

    print("\n--- Quadrature ---")
    print("Recovered total mass:", quad_result.metadata["mass_recovered"])
    print("Real-system residual:", quad_result.metadata["l2_residual_real_system"])
    print("Max abs moment residual:", quad_result.metadata["max_abs_moment_residual"])
    print("L1 CDF error vs true benchmark:", quad_cdf_l1)

    print_cd_peaks(cd_result)
    print_quadrature_atoms(quad_atoms)

    # -------------------------------------------------------------
    # Save arrays
    # -------------------------------------------------------------
    np.savez(
        OUTPUT_DIR / "mixed_all_methods_results.npz",
        moments=moments,
        angles=angles,
        true_ac_density=true_ac_density,
        true_cdf=true_cdf,
        cd_density_proxy=cd_result.density_proxy,
        cd_kernel_diag=cd_result.kernel_diag,
        cesaro_density=cesaro_density,
        cesaro_cdf=cesaro_cdf,
        quad_node_angles=quad_result.node_angles,
        quad_weights=quad_result.weights,
        quad_cdf_grid=quad_result.cdf_grid,
        quad_cdf_values=quad_result.cdf_values,
        quad_cdf_on_angles=quad_cdf_on_angles,
        atomic_angles=ATOMIC_ANGLES,
        atomic_weights=ATOMIC_WEIGHTS,
        scaled_atomic_weights=ATOMIC_MASS * ATOMIC_WEIGHTS,
    )

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------
    plot_density(
        angles=angles,
        cd_density=cd_result.density_proxy,
        cesaro_density=cesaro_density,
        true_ac_density=true_ac_density,
        atomic_angles=ATOMIC_ANGLES,
    )

    plot_cdf(
        angles=angles,
        true_cdf=true_cdf,
        cesaro_cdf=cesaro_cdf,
        quad_cdf=quad_cdf_on_angles,
    )

    recovered_angles = quad_result.node_angles[quad_result.weights > 1e-3]
    recovered_weights = quad_result.weights[quad_result.weights > 1e-3]

    plot_unit_circle_atoms(
        true_atom_angles=ATOMIC_ANGLES,
        true_scaled_weights=ATOMIC_MASS * ATOMIC_WEIGHTS,
        recovered_angles=recovered_angles,
        recovered_weights=recovered_weights,
    )

    plt.show()


if __name__ == "__main__":
    main()