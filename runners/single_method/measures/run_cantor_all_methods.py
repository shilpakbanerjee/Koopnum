from __future__ import annotations

"""
Cantor benchmark for singular-continuous spectral-measure experiments.

Purpose
-------
This runner studies how the current reconstruction pipeline behaves on a
canonical singular-continuous target: the middle-third Cantor measure on
the circle, approximated at finite stage by an equally weighted atomic
measure.

Why this benchmark matters
--------------------------
The finite-stage Cantor approximation is still atomic, but it converges to a
singular-continuous measure as the stage increases. This makes it a useful
bridge between:
    - atomic behavior,
    - singular-continuous limiting behavior,
    - weak approximation methods from moments.

Methods compared
----------------
1. CD-kernel reconstruction:
   Useful as a spectral-detection tool, but not a genuine density in the
   singular-continuous setting.

2. Cesàro / Fejér reconstruction:
   Produces an absolutely continuous weak approximation from moments.

3. Quadrature reconstruction:
   Produces a purely atomic weak approximation from moments.

Key viewpoint
-------------
For singular-continuous measures, the density plot is not the right object.
The CDF and interval-mass statistics are more informative.

Outputs
-------
Plots:
    - cantor_density.png
    - cantor_cdf.png
    - cantor_flatness_vs_scale.png

Data:
    - cantor_results.npz

Printed diagnostics:
    - Cesàro oscillation
    - multiscale CDF slope (included mainly as a cautionary diagnostic)
    - interval-mass intermittency / flatness statistics
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from methods.common.measures.cantor_measure import (
    cantor_measure_on_circle,
    moments_from_atomic,
)
from methods.cd_kernel.measure_reconstruction import (
    evaluate_cd_kernel_from_moments,
)
from methods.common.measures.cesaro import (
    cesaro_density_from_moments,
)
from methods.common.measures.quadrature import (
    reconstruct_atomic_measure_from_moments,
)

Array = np.ndarray

OUTPUT_DIR = Path("experiments/cd_kernel/outputs/measures/cantor")
PLOT_DIR = Path("experiments/cd_kernel/plots/measures/cantor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

STAGE = 10
ORDER = 80
GRID_SIZE = 2048
QUADRATURE_NODE_MULTIPLIER = 6
QUADRATURE_MASS_CONSTRAINT_WEIGHT = 5.0


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def step_cdf_on_grid(step_x: Array, step_y: Array, grid: Array) -> Array:
    """
    Evaluate a right-continuous step CDF on a target grid.

    Parameters
    ----------
    step_x:
        Sorted jump locations.
    step_y:
        Cumulative values at those locations.
    grid:
        Target grid.

    Returns
    -------
    ndarray
        Right-continuous step function evaluated on the grid.
    """
    step_x = np.asarray(step_x, dtype=float)
    step_y = np.asarray(step_y, dtype=float)
    grid = np.asarray(grid, dtype=float)

    idx = np.searchsorted(step_x, grid, side="right") - 1
    out = np.zeros_like(grid, dtype=float)
    mask = idx >= 0
    out[mask] = step_y[idx[mask]]
    return out


def atomic_cdf_on_grid(atom_angles: Array, atom_weights: Array, grid: Array) -> Array:
    """
    Evaluate the true atomic CDF of a finite atomic measure on a target grid.
    """
    atom_angles = np.asarray(atom_angles, dtype=float)
    atom_weights = np.asarray(atom_weights, dtype=float)
    grid = np.asarray(grid, dtype=float)

    order = np.argsort(atom_angles)
    atom_angles = atom_angles[order]
    atom_weights = atom_weights[order]
    cumulative = np.cumsum(atom_weights)

    idx = np.searchsorted(atom_angles, grid, side="right") - 1
    out = np.zeros_like(grid, dtype=float)
    mask = idx >= 0
    out[mask] = cumulative[idx[mask]]
    return out


def multiscale_cdf_slope(cdf: Array, angles: Array, lags: Array | None = None) -> dict:
    """
    Fit log(mean |F(x+h)-F(x)|) vs log(h) over multiple scales.

    Important:
    This diagnostic is included mainly for documentation/comparison.
    Its fitted slope is not, by itself, a good discriminator of
    absolutely continuous vs singular-continuous vs atomic measures,
    because the *mean* interval mass scales linearly in h quite generally.

    # The mean interval-mass slope is not a reliable classifier of spectral type.
    # It is retained here mainly as a cautionary diagnostic and for comparison.
    # Intermittency / flatness is the more informative statistic in this runner.
    """
    cdf = np.asarray(cdf, dtype=float)
    angles = np.asarray(angles, dtype=float)

    if lags is None:
        lags = np.array([1, 2, 4, 8, 16, 32, 64], dtype=int)

    lags = lags[lags < len(cdf) // 4]

    hs = []
    avg_increments = []

    for lag in lags:
        h = angles[lag] - angles[0]
        inc = np.abs(cdf[lag:] - cdf[:-lag])
        avg_inc = np.mean(inc)

        if avg_inc > 1e-14:
            hs.append(h)
            avg_increments.append(avg_inc)

    hs = np.asarray(hs, dtype=float)
    avg_increments = np.asarray(avg_increments, dtype=float)

    if len(hs) < 2:
        raise ValueError("Not enough valid scales for multiscale slope fit")

    log_h = np.log(hs)
    log_inc = np.log(avg_increments)
    slope, intercept = np.polyfit(log_h, log_inc, 1)

    return {
        "hs": hs,
        "avg_increments": avg_increments,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def interval_mass_statistics(cdf: Array, angles: Array, lags: Array | None = None) -> list[dict]:
    """
    Compute interval-mass statistics across scales.

    For each lag, define interval masses
        m_i(h) = F(theta_{i+lag}) - F(theta_i).

    We report:
        - mean mass
        - standard deviation
        - coefficient of variation
        - flatness = E[m^2] / E[m]^2

    This is much more informative than the mean increment slope for
    distinguishing intermittency / concentration of mass.
    """
    cdf = np.asarray(cdf, dtype=float)
    angles = np.asarray(angles, dtype=float)

    if lags is None:
        lags = np.array([1, 2, 4, 8, 16, 32, 64], dtype=int)

    lags = lags[lags < len(cdf) // 4]

    out = []
    for lag in lags:
        h = angles[lag] - angles[0]
        masses = np.abs(cdf[lag:] - cdf[:-lag])

        mean_mass = float(np.mean(masses))
        std_mass = float(np.std(masses))
        second_moment = float(np.mean(masses ** 2))

        cv = std_mass / (mean_mass + 1e-15)
        flatness = second_moment / ((mean_mass + 1e-15) ** 2)

        out.append({
            "lag": int(lag),
            "h": float(h),
            "mean_mass": mean_mass,
            "std_mass": std_mass,
            "cv": float(cv),
            "flatness": float(flatness),
        })

    return out


def print_multiscale_diag(name: str, diag: dict) -> None:
    print(f"\n[{name}]")
    for h, a in zip(diag["hs"], diag["avg_increments"]):
        print(f"h = {h:.6e}, average increment = {a:.6e}")
    print(f"fitted slope ≈ {diag['slope']:.4f}")


def print_interval_stats(name: str, stats: list[dict]) -> None:
    print(f"\n[{name}]")
    for s in stats:
        print(
            f"h = {s['h']:.6e}, "
            f"mean = {s['mean_mass']:.6e}, "
            f"std = {s['std_mass']:.6e}, "
            f"CV = {s['cv']:.6e}, "
            f"flatness = {s['flatness']:.6e}"
        )


def plot_density(cd_angles: Array, cd_density: Array, cesaro_angles: Array, cesaro_density: Array) -> None:
    """
    Plot density-like reconstructions.

    Warning:
    In the singular-continuous setting this plot is heuristic only.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(cd_angles, cd_density, label="CD kernel")
    plt.plot(cesaro_angles, cesaro_density, label="Cesàro")
    plt.title("Cantor measure (singular continuous): density view")
    plt.xlabel("Angle")
    plt.ylabel("Density / proxy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cantor_density.png", dpi=150)


def plot_cdf(grid: Array, true_atomic_cdf: Array, cesaro_angles: Array, cesaro_cdf: Array, quad_cdf: Array) -> None:
    """
    Plot CDF-level comparison, which is the correct viewpoint here.
    """
    plt.figure(figsize=(10, 4))
    plt.step(grid, true_atomic_cdf, where="post", label="True atomic CDF", linewidth=1.2)
    plt.plot(cesaro_angles, cesaro_cdf, label="Cesàro CDF", linewidth=1.5)
    plt.step(grid, quad_cdf, where="post", label="Quadrature CDF", linewidth=1.4)
    plt.title("Cantor measure: CDF comparison")
    plt.xlabel("Angle")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cantor_cdf.png", dpi=150)


def plot_flatness_vs_scale(cesaro_stats: list[dict]) -> None:
    """
    Plot flatness of interval masses vs scale for the Cesàro reconstruction.
    """
    hs = np.array([s["h"] for s in cesaro_stats], dtype=float)
    flatness = np.array([s["flatness"] for s in cesaro_stats], dtype=float)

    plt.figure(figsize=(6.5, 4.8))
    plt.loglog(hs, flatness, marker="o")
    plt.title("Flatness vs scale (Cesàro)")
    plt.xlabel("Scale h")
    plt.ylabel("Flatness")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cantor_flatness_vs_scale.png", dpi=150)


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def main() -> None:
    # -------------------------------------------------------------
    # Construct finite-stage Cantor approximation and its moments
    # -------------------------------------------------------------
    true_atom_angles, true_atom_weights = cantor_measure_on_circle(STAGE)
    moments = moments_from_atomic(
        angles=true_atom_angles,
        weights=true_atom_weights,
        order=ORDER,
    )

    # -------------------------------------------------------------
    # Reconstruct from moments using the three methods
    # -------------------------------------------------------------
    cd_result = evaluate_cd_kernel_from_moments(
        moments,
        order=ORDER,
        grid_size=GRID_SIZE,
        normalize_density=True,
    )

    cesaro_result = cesaro_density_from_moments(
        moments,
        order=ORDER,
        grid_size=GRID_SIZE,
        clip_negative=True,
        normalize_mass=True,
    )

    quad_result = reconstruct_atomic_measure_from_moments(
        moments,
        order=ORDER,
        num_nodes=QUADRATURE_NODE_MULTIPLIER * (ORDER + 1),
        mass_constraint_weight=QUADRATURE_MASS_CONSTRAINT_WEIGHT,
        normalize_mass=True,
    )

    # -------------------------------------------------------------
    # Put everything on a common grid for comparison
    # -------------------------------------------------------------
    grid = cd_result.angles

    true_atomic_cdf = atomic_cdf_on_grid(
        true_atom_angles,
        true_atom_weights,
        grid,
    )

    quad_cdf_on_grid = step_cdf_on_grid(
        quad_result.cdf_grid,
        quad_result.cdf_values,
        grid,
    )

    # -------------------------------------------------------------
    # Basic oscillation diagnostic for Cesàro "density"
    # -------------------------------------------------------------
    cesaro_density = np.asarray(cesaro_result.density, dtype=float)
    osc = np.max(np.abs(np.diff(cesaro_density)))
    osc_norm = osc / (np.mean(cesaro_density) + 1e-12)

    print("\n--- Cesàro oscillation diagnostic ---")
    print(f"max |Δ rho| = {osc:.6e}")
    print(f"normalized oscillation = {osc_norm:.6e}")

    # -------------------------------------------------------------
    # Multiscale average-increment slope
    # Included mainly as a cautionary reference diagnostic
    # -------------------------------------------------------------
    true_diag = multiscale_cdf_slope(true_atomic_cdf, grid)
    cesaro_diag = multiscale_cdf_slope(cesaro_result.cdf, cesaro_result.angles)
    quad_diag = multiscale_cdf_slope(quad_cdf_on_grid, grid)

    print("\n--- Multiscale CDF scaling diagnostic ---")
    print_multiscale_diag("True atomic CDF", true_diag)
    print_multiscale_diag("Cesàro CDF", cesaro_diag)
    print_multiscale_diag("Quadrature CDF", quad_diag)

    # -------------------------------------------------------------
    # Interval-mass intermittency / flatness
    # This is the more informative diagnostic here.
    # -------------------------------------------------------------
    true_stats = interval_mass_statistics(true_atomic_cdf, grid)
    cesaro_stats = interval_mass_statistics(cesaro_result.cdf, cesaro_result.angles)
    quad_stats = interval_mass_statistics(quad_cdf_on_grid, grid)

    print("\n--- Interval-mass intermittency diagnostic ---")
    print_interval_stats("True atomic CDF", true_stats)
    print_interval_stats("Cesàro CDF", cesaro_stats)
    print_interval_stats("Quadrature CDF", quad_stats)

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------
    plot_density(
        cd_angles=grid,
        cd_density=cd_result.density_proxy,
        cesaro_angles=cesaro_result.angles,
        cesaro_density=cesaro_result.density,
    )

    plot_cdf(
        grid=grid,
        true_atomic_cdf=true_atomic_cdf,
        cesaro_angles=cesaro_result.angles,
        cesaro_cdf=cesaro_result.cdf,
        quad_cdf=quad_cdf_on_grid,
    )

    plot_flatness_vs_scale(cesaro_stats)

    plt.show()

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------
    np.savez(
        OUTPUT_DIR / "cantor_results.npz",
        moments=moments,
        true_atom_angles=true_atom_angles,
        true_atom_weights=true_atom_weights,
        true_atomic_cdf=true_atomic_cdf,
        cd_angles=grid,
        cd_density=cd_result.density_proxy,
        cesaro_angles=cesaro_result.angles,
        cesaro_density=cesaro_result.density,
        cesaro_cdf=cesaro_result.cdf,
        quad_nodes=quad_result.node_angles,
        quad_weights=quad_result.weights,
        quad_cdf_grid=quad_result.cdf_grid,
        quad_cdf_values=quad_result.cdf_values,
        quad_cdf_on_grid=quad_cdf_on_grid,
        cesaro_osc=osc,
        cesaro_osc_norm=osc_norm,
    )

    print("\nDone: Cantor benchmark")
    print(f"Stage = {STAGE}, atoms = {2 ** STAGE}")


if __name__ == "__main__":
    main()