"""
Compare three moment-based reconstruction methods on the dynamical side.

Methods compared
----------------
1. Baseline CD-kernel reconstruction from empirical moments
2. Cesàro / Fejér weak reconstruction from the same empirical moments
3. Quadrature / atomic weak reconstruction from the same empirical moments

Pipeline
--------
system -> trajectory -> observable signal -> empirical moments
      -> CD kernel
      -> Cesàro
      -> quadrature

This runner is meant to complement the measure-side comparison runner by
testing all three methods on moments estimated from actual trajectories.

Suggested use
-------------
Start with CASE = "rotation" for a clean pure-point benchmark.
Then try CASE = "catmap" for a predominantly continuous-spectrum benchmark.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.core.diagnostics import (
    summarize_result,
    compare_results,
    weak_convergence_summary,
)
from experiments.cd_kernel.dynamics.systems import (
    generate_planar_rotation,
    generate_cat_map,
)
from experiments.cd_kernel.dynamics.observables import (
    complex_coordinate,
    torus_fourier_mode,
)
from experiments.cd_kernel.dynamics.spectral_measure import (
    reconstruct_spectral_measure_from_system,
)
from experiments.cd_kernel.measures.cesaro import cesaro_density_from_moments
from experiments.cd_kernel.measures.quadrature import (
    reconstruct_atomic_measure_from_moments,
    significant_atoms,
)
from experiments.cd_kernel.runners.common_plotting import (
    save_density_comparison_plot,
    save_density_comparison_log_plot,
    save_density_comparison_normalized_plot,
    save_peak_overlay_plot,
    save_difference_plot,
)
from experiments.cd_kernel.core.peaks import find_top_peaks

Array = np.ndarray


# ---------------------------------------------------------------------
# Choose benchmark here
# ---------------------------------------------------------------------
CASE = "rotation"   # "rotation" or "catmap"


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


@dataclass
class WrappedDensityResult:
    angles: Array
    density_proxy: Array
    kernel_diag: Array
    circle_points: Array
    metadata: dict

    def top_peaks(self, k: int = 10, min_separation: int = 8):
        idx = find_top_peaks(
            self.density_proxy,
            k=k,
            min_separation=min_separation,
        )
        return [
            {
                "index": int(i),
                "angle": float(self.angles[i]),
                "point": complex(self.circle_points[i]),
                "value": float(self.density_proxy[i]),
            }
            for i in idx
        ]


def result_like_from_density(
    angles: Array,
    density: Array,
    label: str,
    metadata: dict | None = None,
) -> WrappedDensityResult:
    """
    Wrap a density on the unit-circle grid into a result-like object
    compatible with diagnostics and plotting utilities.
    """
    angles = np.asarray(angles, dtype=float)
    density = np.asarray(density, dtype=float)
    circle_points = np.exp(1j * angles)

    if metadata is None:
        metadata = {}
    metadata = dict(metadata)
    metadata["variant"] = label
    metadata["source"] = "trajectory"

    return WrappedDensityResult(
        angles=angles,
        density_proxy=density,
        kernel_diag=density.copy(),
        circle_points=circle_points,
        metadata=metadata,
    )


def cdf_from_density(density: Array, angles: Array) -> Array:
    density = np.asarray(density, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if len(angles) < 2:
        return np.zeros_like(angles)
    dtheta = float(angles[1] - angles[0])
    return np.cumsum(density) * dtheta


def circle_integral(values: Array, angles: Array) -> float:
    values = np.asarray(values, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, angles))
    return float(np.trapz(values, angles))


def l1_error(y_a: Array, y_b: Array, x: Array) -> float:
    return circle_integral(np.abs(np.asarray(y_a) - np.asarray(y_b)), x)


def case_config(case: str) -> dict:
    """
    Centralized benchmark configuration.
    """
    case = case.lower()

    if case == "rotation":
        return {
            "name": "rotation",
            "title": "Planar rotation: CD vs Cesàro vs quadrature",
            "output_slug": "rotation_all_methods",
            "system_fn": generate_planar_rotation,
            "system_kwargs": {
                "n": 2500,
                "theta": 0.35,
            },
            "observable": complex_coordinate(0, 1),
            "order": 80,
            "grid_size": 2048,
            "regularization": 1e-6,
            "center": False,
            "normalize_moments": True,
            "taper": None,
            "quadrature_num_nodes": 10 * (80 + 1),
            "quadrature_mass_constraint_weight": 5.0,
            "known_atom_angles": np.array([0.35], dtype=float),
            "notes": "Pure-point benchmark.",
        }

    if case == "catmap":
        return {
            "name": "catmap",
            "title": "Cat map: CD vs Cesàro vs quadrature",
            "output_slug": "catmap_all_methods",
            "system_fn": generate_cat_map,
            "system_kwargs": {
                "n": 5000,
            },
            "observable": torus_fourier_mode(1, 0),
            "order": 120,
            "grid_size": 2048,
            "regularization": 1e-8,
            "center": True,
            "normalize_moments": True,
            "taper": None,
            "quadrature_num_nodes": 10 * (120 + 1),
            "quadrature_mass_constraint_weight": 3.0,
            "known_atom_angles": None,
            "notes": "Predominantly continuous-spectrum benchmark.",
        }

    raise ValueError(f"Unknown CASE '{case}'. Use 'rotation' or 'catmap'.")


def main():
    cfg = case_config(CASE)

    output_dir = Path(f"experiments/cd_kernel/outputs/dynamics/{cfg['output_slug']}")
    plot_dir = Path(f"experiments/cd_kernel/plots/dynamics/{cfg['output_slug']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Trajectory -> moments -> baseline CD-kernel
    # ---------------------------------------------------------------
    X, spec_data, cd_result = reconstruct_spectral_measure_from_system(
        system_fn=cfg["system_fn"],
        system_kwargs=cfg["system_kwargs"],
        order=cfg["order"],
        observable=cfg["observable"],
        center=cfg["center"],
        normalize_moments=cfg["normalize_moments"],
        taper=cfg["taper"],
        grid_size=cfg["grid_size"],
        regularization=cfg["regularization"],
        normalize_density=True,
    )
    cd_result.metadata["variant"] = "cd_kernel_v001_baseline"

    moments = spec_data.moments
    angles = cd_result.angles

    # ---------------------------------------------------------------
    # Cesàro from the same empirical moments
    # ---------------------------------------------------------------
    cesaro_data = cesaro_density_from_moments(
        moments=moments,
        order=cfg["order"],
        grid_size=cfg["grid_size"],
        clip_negative=True,
        normalize_mass=True,
    )
    cesaro_result = result_like_from_density(
        angles=cesaro_data.angles,
        density=cesaro_data.density,
        label="cesaro_fejer",
        metadata={
            "method": "cesaro_fejer",
            "signal_length": spec_data.metadata["signal_length"],
            "case": cfg["name"],
        },
    )

    # ---------------------------------------------------------------
    # Quadrature from the same empirical moments
    # ---------------------------------------------------------------
    quad_data = reconstruct_atomic_measure_from_moments(
        moments=moments,
        order=cfg["order"],
        num_nodes=cfg["quadrature_num_nodes"],
        mass_constraint_weight=cfg["quadrature_mass_constraint_weight"],
        normalize_mass=True,
    )

    quad_cdf_on_angles = step_cdf_on_grid(
        quad_data.cdf_grid,
        quad_data.cdf_values,
        angles,
    )

    # Convert step CDF to a grid density proxy for plotting/diagnostics.
    # This is not a smooth density; it is only a visualized atomic proxy.
    quad_density_proxy = np.gradient(quad_cdf_on_angles, angles[1] - angles[0])
    quad_density_proxy = np.maximum(np.real(quad_density_proxy), 0.0)

    quad_result = result_like_from_density(
        angles=angles,
        density=quad_density_proxy,
        label="quadrature_atomic_proxy",
        metadata={
            "method": "quadrature_uniform_grid",
            "signal_length": spec_data.metadata["signal_length"],
            "case": cfg["name"],
            "l2_residual_real_system": quad_data.metadata["l2_residual_real_system"],
            "max_abs_moment_residual": quad_data.metadata["max_abs_moment_residual"],
            "num_nodes": quad_data.metadata["num_nodes"],
        },
    )

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------
    cd_summary = summarize_result(cd_result)
    cesaro_summary = summarize_result(cesaro_result)
    quad_summary = summarize_result(quad_result)

    cd_vs_cesaro = compare_results(cd_result, cesaro_result)
    cd_vs_quad = compare_results(cd_result, quad_result)
    cesaro_vs_quad = compare_results(cesaro_result, quad_result)

    weak_cd_vs_cesaro = weak_convergence_summary(cd_result, cesaro_result, max_mode=12)
    weak_cd_vs_quad = weak_convergence_summary(cd_result, quad_result, max_mode=12)
    weak_cesaro_vs_quad = weak_convergence_summary(cesaro_result, quad_result, max_mode=12)

    cd_cdf = cdf_from_density(cd_result.density_proxy, angles)
    cesaro_cdf = np.asarray(cesaro_data.cdf, dtype=float)
    cdf_cd_vs_cesaro = l1_error(cd_cdf, cesaro_cdf, angles)
    cdf_cd_vs_quad = l1_error(cd_cdf, quad_cdf_on_angles, angles)
    cdf_cesaro_vs_quad = l1_error(cesaro_cdf, quad_cdf_on_angles, angles)

    recovered_atoms = significant_atoms(quad_data, tol=1e-3)

    print(f"\n=== {cfg['title']} ===")
    print(cfg["notes"])
    print(f"order = {cfg['order']}")
    print(f"signal_length = {spec_data.metadata['signal_length']}")
    print()

    print("--- CD summary ---")
    for k, v in cd_summary.items():
        print(f"  {k}: {v:.6e}")

    print("\n--- Cesàro summary ---")
    for k, v in cesaro_summary.items():
        print(f"  {k}: {v:.6e}")

    print("\n--- Quadrature summary ---")
    for k, v in quad_summary.items():
        print(f"  {k}: {v:.6e}")

    print("\n--- Pairwise density diagnostics ---")
    print("CD vs Cesàro:")
    for k, v in cd_vs_cesaro.items():
        print(f"  {k}: {v:.6e}")

    print("\nCD vs Quadrature:")
    for k, v in cd_vs_quad.items():
        print(f"  {k}: {v:.6e}")

    print("\nCesàro vs Quadrature:")
    for k, v in cesaro_vs_quad.items():
        print(f"  {k}: {v:.6e}")

    print("\n--- Pairwise weak diagnostics ---")
    print("CD vs Cesàro:")
    for k, v in weak_cd_vs_cesaro.items():
        print(f"  {k}: {v:.6e}")

    print("\nCD vs Quadrature:")
    for k, v in weak_cd_vs_quad.items():
        print(f"  {k}: {v:.6e}")

    print("\nCesàro vs Quadrature:")
    for k, v in weak_cesaro_vs_quad.items():
        print(f"  {k}: {v:.6e}")

    print("\n--- Pairwise CDF-level L1 discrepancies ---")
    print(f"  L1(CDF_CD, CDF_Cesàro)       = {cdf_cd_vs_cesaro:.6e}")
    print(f"  L1(CDF_CD, CDF_Quadrature)   = {cdf_cd_vs_quad:.6e}")
    print(f"  L1(CDF_Cesàro, CDF_Quadrature)= {cdf_cesaro_vs_quad:.6e}")

    print("\n--- Quadrature residuals ---")
    print(f"  recovered total mass = {quad_data.metadata['mass_recovered']:.6e}")
    print(f"  real-system residual = {quad_data.metadata['l2_residual_real_system']:.6e}")
    print(f"  max abs moment residual = {quad_data.metadata['max_abs_moment_residual']:.6e}")

    print("\n--- Quadrature significant atoms (weight > 1e-3) ---")
    for atom in recovered_atoms[:25]:
        print(
            f"  angle={atom['angle']:.6f}, "
            f"weight={atom['weight']:.6e}, "
            f"point={atom['point']}"
        )

    if cfg["known_atom_angles"] is not None:
        print("\n--- Known reference atom angles ---")
        for theta in cfg["known_atom_angles"]:
            print(f"  theta = {theta:.6f}")

    print("\n--- CD top peaks ---")
    for item in cd_result.top_peaks(k=10, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )

    print("\n--- Cesàro top peaks ---")
    for item in cesaro_result.top_peaks(k=10, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )

    print("\n--- Quadrature proxy top peaks ---")
    for item in quad_result.top_peaks(k=10, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )

    # ---------------------------------------------------------------
    # Save arrays
    # ---------------------------------------------------------------
    np.savez(
        output_dir / f"{cfg['name']}_all_methods_results.npz",
        trajectory=X,
        signal=spec_data.signal,
        moments=moments,
        cd_angles=cd_result.angles,
        cd_density=cd_result.density_proxy,
        cd_kernel=cd_result.kernel_diag,
        cesaro_angles=cesaro_data.angles,
        cesaro_density=cesaro_data.density,
        cesaro_cdf=cesaro_data.cdf,
        quad_node_angles=quad_data.node_angles,
        quad_weights=quad_data.weights,
        quad_cdf_grid=quad_data.cdf_grid,
        quad_cdf_values=quad_data.cdf_values,
        quad_cdf_on_angles=quad_cdf_on_angles,
        quad_density_proxy=quad_density_proxy,
        cd_summary=np.array(list(cd_summary.items()), dtype=object),
        cesaro_summary=np.array(list(cesaro_summary.items()), dtype=object),
        quad_summary=np.array(list(quad_summary.items()), dtype=object),
        cd_vs_cesaro=np.array(list(cd_vs_cesaro.items()), dtype=object),
        cd_vs_quad=np.array(list(cd_vs_quad.items()), dtype=object),
        cesaro_vs_quad=np.array(list(cesaro_vs_quad.items()), dtype=object),
        weak_cd_vs_cesaro=np.array(list(weak_cd_vs_cesaro.items()), dtype=object),
        weak_cd_vs_quad=np.array(list(weak_cd_vs_quad.items()), dtype=object),
        weak_cesaro_vs_quad=np.array(list(weak_cesaro_vs_quad.items()), dtype=object),
    )

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    styles = [
        {"linestyle": "-", "alpha": 0.85, "linewidth": 1.6},
        {"linestyle": "--", "alpha": 0.92, "linewidth": 1.6},
        {"linestyle": ":", "alpha": 0.92, "linewidth": 1.8},
    ]

    save_density_comparison_plot(
        results=[cd_result, cesaro_result, quad_result],
        labels=["CD kernel", "Cesàro", "Quadrature proxy"],
        title=cfg["title"],
        save_path=plot_dir / f"{cfg['name']}_all_methods_density.png",
        show_peaks=True,
        styles=styles,
    )

    save_density_comparison_log_plot(
        results=[cd_result, cesaro_result, quad_result],
        labels=["CD kernel", "Cesàro", "Quadrature proxy"],
        title=f"{cfg['title']} (log scale)",
        save_path=plot_dir / f"{cfg['name']}_all_methods_density_log.png",
        styles=styles,
    )

    save_density_comparison_normalized_plot(
        results=[cd_result, cesaro_result, quad_result],
        labels=["CD / max", "Cesàro / max", "Quadrature proxy / max"],
        title=f"{cfg['title']} (max-normalized)",
        save_path=plot_dir / f"{cfg['name']}_all_methods_density_normalized.png",
        styles=styles,
    )

    save_peak_overlay_plot(
        base_result=cesaro_result,
        overlay_results=[cd_result, cesaro_result, quad_result],
        overlay_labels=["CD peaks", "Cesàro peaks", "Quadrature proxy peaks"],
        title=f"{cfg['title']}: detected peaks",
        save_path=plot_dir / f"{cfg['name']}_all_methods_peaks.png",
        base_label="Cesàro density",
    )

    save_difference_plot(
        result_a=cd_result,
        result_b=cesaro_result,
        title=f"{cfg['title']}: |CD - Cesàro|",
        save_path=plot_dir / f"{cfg['name']}_cd_minus_cesaro.png",
    )

    save_difference_plot(
        result_a=cd_result,
        result_b=quad_result,
        title=f"{cfg['title']}: |CD - Quadrature proxy|",
        save_path=plot_dir / f"{cfg['name']}_cd_minus_quadrature.png",
    )

    # ---------------------------------------------------------------
    # CDF comparison plot
    # ---------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10.5, 5.0))
    ax1.plot(angles, cd_cdf, linewidth=1.6, label="CD-derived CDF")
    ax1.plot(angles, cesaro_cdf, linewidth=1.6, label="Cesàro CDF")
    ax1.step(
        angles,
        quad_cdf_on_angles,
        where="post",
        linewidth=1.4,
        label="Quadrature CDF",
    )
    if cfg["known_atom_angles"] is not None:
        for j, theta in enumerate(cfg["known_atom_angles"]):
            ax1.axvline(
                theta,
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
                label="Known atom angle" if j == 0 else None,
            )
    ax1.set_xlabel("Angle on unit circle")
    ax1.set_ylabel("CDF")
    ax1.set_title(f"{cfg['title']}: CDF-level comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(plot_dir / f"{cfg['name']}_all_methods_cdf.png", dpi=160)

    # ---------------------------------------------------------------
    # Unit-circle quadrature atoms
    # ---------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
    theta_circle = np.linspace(0.0, 2.0 * np.pi, 800)
    ax2.plot(np.cos(theta_circle), np.sin(theta_circle), linewidth=1.0, color="black")

    if cfg["known_atom_angles"] is not None:
        ax2.scatter(
            np.cos(cfg["known_atom_angles"]),
            np.sin(cfg["known_atom_angles"]),
            s=200.0,
            marker="o",
            label="Known atoms",
        )

    keep = quad_data.weights > 1e-3
    recovered_angles = quad_data.node_angles[keep]
    recovered_weights = quad_data.weights[keep]
    if len(recovered_angles) > 0:
        recovered_sizes = 400.0 * recovered_weights / np.max(recovered_weights)
        ax2.scatter(
            np.cos(recovered_angles),
            np.sin(recovered_angles),
            s=recovered_sizes,
            marker="x",
            label="Quadrature atoms",
        )

    ax2.set_aspect("equal")
    ax2.set_title(f"{cfg['title']}: quadrature atoms on unit circle")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(plot_dir / f"{cfg['name']}_all_methods_unit_circle_atoms.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()