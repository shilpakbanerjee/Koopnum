"""
Compare three moment-based reconstruction methods on a mixed benchmark measure.

Methods compared
----------------
1. Baseline CD-kernel density proxy
2. Cesàro / Fejér weak reconstruction (Section 4.2.1 of Korda–Putinar–Mezić)
3. Quadrature / atomic weak reconstruction (Section 4.2.2, uniform-grid practical version)

Benchmark
---------
We use a mixed measure on the unit circle of the form

    mu = w_ac * mu_ac + w_at * mu_at,

where mu_ac is absolutely continuous and mu_at is atomic.

This runner is designed to answer:
- How does the CD density proxy behave on a mixed spectrum?
- Does the Cesàro reconstruction recover the absolutely continuous contribution at CDF level?
- Does the quadrature reconstruction recover the atomic structure / weak CDF?

Outputs
-------
- numerical arrays saved to .npz
- density comparison plot
- CDF comparison plot
- unit-circle atom comparison plot
- simple printed diagnostics

Notes
-----
- The CD-kernel output is a density proxy, not a weakly convergent measure reconstruction.
- The Cesàro method yields an absolutely continuous approximating measure.
- The quadrature method yields a purely atomic approximating measure.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.measures.benchmark_measures import (
    AtomicMeasure,
    AbsolutelyContinuousMeasure,
    wrapped_gaussian_density,
)
from experiments.cd_kernel.measures.moments import normalize_moments
from experiments.cd_kernel.measures.reconstruction import evaluate_cd_kernel_from_moments
from experiments.cd_kernel.measures.cesaro import cesaro_density_from_moments
from experiments.cd_kernel.measures.quadrature import (
    reconstruct_atomic_measure_from_moments,
    significant_atoms,
)

Array = np.ndarray

OUTPUT_DIR = Path("experiments/cd_kernel/outputs/measures/mixed_all_methods")
PLOT_DIR = Path("experiments/cd_kernel/plots/measures/mixed_all_methods")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


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
    Smooth absolutely continuous density on [0, 2pi), normalized to integrate to 1.
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
    Build normalized moments of a mixed measure:
        mu = ac_mass * mu_ac + atomic_mass * mu_at
    """
    mu_at = AtomicMeasure(angles=np.asarray(atomic_angles), weights=np.asarray(atomic_weights))
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
    Right-continuous CDF on the supplied angular grid.
    """
    rho_ac = exact_true_ac_density(angles, ac_density_fn=ac_density_fn, ac_mass=ac_mass)
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
    Evaluate a right-continuous step CDF given by node locations step_x and cumulative
    values step_y on a target grid.
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


def main():
    # Reconstruction parameters
    order = 80
    grid_size = 2048
    moment_grid_size = 4096
    regularization = 1e-8

    # Mixed benchmark: atomic + AC
    atomic_angles = np.array([0.55, 2.15, 5.00], dtype=float)
    atomic_weights = np.array([0.18, 0.12, 0.10], dtype=float)
    atomic_weights = atomic_weights / np.sum(atomic_weights)

    atomic_mass = 0.40
    ac_mass = 0.60

    ac_density_fn = mixed_ac_density(
        weight1=0.58,
        center1=1.35,
        sigma1=0.22,
        weight2=0.42,
        center2=4.55,
        sigma2=0.36,
    )

    # Exact benchmark moments
    moments = exact_mixed_moments(
        order=order,
        atomic_angles=atomic_angles,
        atomic_weights=atomic_weights,
        ac_density_fn=ac_density_fn,
        atomic_mass=atomic_mass,
        ac_mass=ac_mass,
        ac_grid_size=moment_grid_size,
    )

    # Method 1: baseline CD-kernel density proxy
    cd_result = evaluate_cd_kernel_from_moments(
        moments,
        order=order,
        grid_size=grid_size,
        regularization=regularization,
        normalize_density=True,
    )

    # Method 2: Cesàro / Fejér reconstruction
    cesaro_result = cesaro_density_from_moments(
        moments,
        order=order,
        grid_size=grid_size,
        clip_negative=True,
        normalize_mass=True,
    )

    # Method 3: Quadrature / atomic weak reconstruction
    quad_result = reconstruct_atomic_measure_from_moments(
        moments,
        order=order,
        num_nodes=6 * (order + 1),
        mass_constraint_weight=5.0,
        normalize_mass=True,
    )

    # Exact benchmark reference on the same grid
    angles = cd_result.angles
    true_ac_density = exact_true_ac_density(
        angles,
        ac_density_fn=ac_density_fn,
        ac_mass=ac_mass,
    )
    true_cdf = exact_true_cdf(
        angles,
        atomic_angles=atomic_angles,
        atomic_weights=atomic_weights,
        ac_density_fn=ac_density_fn,
        atomic_mass=atomic_mass,
        ac_mass=ac_mass,
    )

    # Put Cesàro and quadrature CDFs on the same grid for fair comparison
    cesaro_density = np.asarray(cesaro_result.density, dtype=float)
    cesaro_cdf = np.asarray(cesaro_result.cdf, dtype=float)

    quad_cdf_on_angles = step_cdf_on_grid(
        quad_result.cdf_grid,
        quad_result.cdf_values,
        angles,
    )

    # Diagnostics
    cesaro_cdf_l1 = l1_error(true_cdf, cesaro_cdf, angles)
    quad_cdf_l1 = l1_error(true_cdf, quad_cdf_on_angles, angles)

    quad_atoms = significant_atoms(quad_result, tol=1e-3)

    print("\n=== Mixed measure all-method comparison ===")
    print(f"order = {order}")
    print(f"grid_size = {grid_size}")
    print(f"atomic_mass = {atomic_mass:.3f}, ac_mass = {ac_mass:.3f}")
    print("true atomic angles =", atomic_angles)
    print("true atomic weights (scaled) =", atomic_mass * atomic_weights)
    print()
    print("--- CD kernel ---")
    print("Toeplitz condition number:", cd_result.metadata["toeplitz_condition_number"])
    print("Top peaks:")
    for item in cd_result.top_peaks(k=8, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )
    print()
    print("--- Cesàro ---")
    print("Approximate final CDF value:", float(cesaro_cdf[-1]))
    print("L1 CDF error vs true benchmark:", cesaro_cdf_l1)
    print()
    print("--- Quadrature ---")
    print("Recovered total mass:", quad_result.metadata["mass_recovered"])
    print("Real-system residual:", quad_result.metadata["l2_residual_real_system"])
    print("Max abs moment residual:", quad_result.metadata["max_abs_moment_residual"])
    print("L1 CDF error vs true benchmark:", quad_cdf_l1)
    print("Significant recovered atoms (weight > 1e-3):")
    for atom in quad_atoms[:20]:
        print(
            f"  angle={atom['angle']:.6f}, "
            f"weight={atom['weight']:.6e}, "
            f"point={atom['point']}"
        )

    # Save data
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
        atomic_angles=atomic_angles,
        atomic_weights=atomic_weights,
        scaled_atomic_weights=atomic_mass * atomic_weights,
    )

    # ------------------------------------------------------------------
    # Plot 1: Density comparison
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10.5, 5.0))
    ax1.plot(angles, cd_result.density_proxy, linewidth=1.6, label="CD density proxy")
    ax1.plot(angles, cesaro_density, linewidth=1.6, label="Cesàro density")
    ax1.plot(
        angles,
        true_ac_density,
        linestyle="--",
        linewidth=1.4,
        label="True AC density component",
    )
    for j, theta in enumerate(atomic_angles):
        ax1.axvline(
            theta,
            linestyle=":",
            linewidth=1.0,
            alpha=0.9,
            label="True atom location" if j == 0 else None,
        )
    ax1.set_xlabel("Angle on unit circle")
    ax1.set_ylabel("Density / density proxy")
    ax1.set_title("Mixed benchmark: CD proxy vs Cesàro density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(PLOT_DIR / "mixed_all_methods_density.png", dpi=160)

    # ------------------------------------------------------------------
    # Plot 2: CDF comparison
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10.5, 5.0))
    ax2.plot(angles, true_cdf, linewidth=1.8, label="True CDF")
    ax2.plot(angles, cesaro_cdf, linewidth=1.6, label="Cesàro CDF")
    ax2.step(
        angles,
        quad_cdf_on_angles,
        where="post",
        linewidth=1.3,
        label="Quadrature CDF",
    )
    ax2.set_xlabel("Angle on unit circle")
    ax2.set_ylabel("CDF")
    ax2.set_title("Mixed benchmark: weak reconstructions at CDF level")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "mixed_all_methods_cdf.png", dpi=160)

    # ------------------------------------------------------------------
    # Plot 3: Unit-circle atom comparison
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(6.5, 6.5))
    theta_circle = np.linspace(0.0, 2.0 * np.pi, 800)
    ax3.plot(np.cos(theta_circle), np.sin(theta_circle), linewidth=1.0, color="black")

    # true atoms
    true_scaled_weights = atomic_mass * atomic_weights
    true_sizes = 800.0 * true_scaled_weights / np.max(true_scaled_weights)
    ax3.scatter(
        np.cos(atomic_angles),
        np.sin(atomic_angles),
        s=true_sizes,
        marker="o",
        label="True atoms",
    )

    # recovered quadrature atoms
    recovered_angles = quad_result.node_angles[quad_result.weights > 1e-3]
    recovered_weights = quad_result.weights[quad_result.weights > 1e-3]
    if len(recovered_angles) > 0:
        recovered_sizes = 500.0 * recovered_weights / np.max(recovered_weights)
        ax3.scatter(
            np.cos(recovered_angles),
            np.sin(recovered_angles),
            s=recovered_sizes,
            marker="x",
            label="Quadrature atoms",
        )

    ax3.set_aspect("equal")
    ax3.set_title("True atoms vs quadrature-recovered atoms")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(PLOT_DIR / "mixed_all_methods_unit_circle_atoms.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()