"""
Convergence / stability study for baseline vs tapered CD-kernel
reconstructions on the Arnold cat map.

Unlike the rotation exact-vs-empirical experiment, there is no exact
reference measure used here. Instead, this runner studies:

- how baseline and tapered variants differ
- how that difference changes with trajectory length n
- how that difference changes with moment order

Diagnostics recorded:
- shape diagnostics (L1/L2, flatness, entropy, peak-mass ratio)
- weak-convergence diagnostics based on Fourier test functions

This is intended as a first quantitative study of the continuous-spectrum
regime, where tapering should typically improve stability and reduce
spurious oscillations.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.core.diagnostics import (
    summarize_result,
    compare_results,
    weak_convergence_summary,
)
from experiments.cd_kernel.dynamics.systems import generate_cat_map
from experiments.cd_kernel.dynamics.observables import torus_fourier_mode
from experiments.cd_kernel.dynamics.spectral_measure import (
    reconstruct_spectral_measure_from_system,
)
from experiments.cd_kernel.variants.cd_kernel_v002_tapered import (
    fit_cd_kernel_tapered_from_moments,
)


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/convergence/catmap_variant_convergence")
PLOT_DIR = Path("experiments/cd_kernel/plots/convergence/catmap_variant_convergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def metric_matrix_to_heatmap(
    n_values,
    order_values,
    values,
    title: str,
    save_path: Path,
    cmap: str = "viridis",
):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap)

    ax.set_xticks(np.arange(len(order_values)))
    ax.set_xticklabels(order_values)
    ax.set_yticks(np.arange(len(n_values)))
    ax.set_yticklabels(n_values)

    ax.set_xlabel("Moment order")
    ax.set_ylabel("Trajectory length n")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=90)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def main():
    grid_size = 2048
    regularization = 1e-6

    n_values = [1000, 2000, 4000, 8000]
    order_values = [10, 20, 40, 60, 80, 100]

    # Natural torus observable
    observable = torus_fourier_mode(1, 0)

    # Matrices indexed by [i_n, i_order]
    l1_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    l2_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    weak_fourier_max_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    weak_fourier_mean_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    baseline_flatness_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    tapered_flatness_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    baseline_entropy_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    tapered_entropy_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    baseline_tv_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    tapered_tv_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    baseline_peak_mass_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    tapered_peak_mass_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    baseline_peak_height_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    tapered_peak_height_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    records = []

    for i_n, n in enumerate(n_values):
        print(f"\n=== n = {n} ===")

        for i_o, order in enumerate(order_values):
            print(f"  order = {order}")

            X, spec, baseline = reconstruct_spectral_measure_from_system(
                system_fn=generate_cat_map,
                system_kwargs={"n": n},
                order=order,
                observable=observable,
                center=True,
                normalize_moments=True,
                taper=None,
                grid_size=grid_size,
                regularization=regularization,
                normalize_density=True,
            )
            baseline.metadata["variant"] = "cd_kernel_v001_baseline"

            tapered = fit_cd_kernel_tapered_from_moments(
                moments=spec.moments,
                order=order,
                grid_size=grid_size,
                regularization=regularization,
                taper="fejer",
                normalize_density=True,
            )
            tapered.metadata["variant"] = "cd_kernel_v002_tapered"

            base_sum = summarize_result(baseline)
            tap_sum = summarize_result(tapered)
            comp = compare_results(baseline, tapered)
            weak = weak_convergence_summary(baseline, tapered, max_mode=12)

            l1_mat[i_n, i_o] = comp["l1_distance"]
            l2_mat[i_n, i_o] = comp["l2_distance"]

            weak_fourier_max_mat[i_n, i_o] = weak["fourier_max_abs_discrepancy"]
            weak_fourier_mean_mat[i_n, i_o] = weak["fourier_mean_abs_discrepancy"]

            baseline_flatness_mat[i_n, i_o] = base_sum["spectral_flatness"]
            tapered_flatness_mat[i_n, i_o] = tap_sum["spectral_flatness"]

            baseline_entropy_mat[i_n, i_o] = base_sum["entropy"]
            tapered_entropy_mat[i_n, i_o] = tap_sum["entropy"]

            baseline_tv_mat[i_n, i_o] = base_sum["total_variation"]
            tapered_tv_mat[i_n, i_o] = tap_sum["total_variation"]

            baseline_peak_mass_mat[i_n, i_o] = base_sum["peak_mass_ratio"]
            tapered_peak_mass_mat[i_n, i_o] = tap_sum["peak_mass_ratio"]

            baseline_peak_height_mat[i_n, i_o] = base_sum["max_peak_height"]
            tapered_peak_height_mat[i_n, i_o] = tap_sum["max_peak_height"]

            records.append(
                {
                    "n": n,
                    "order": order,
                    "baseline_summary": base_sum,
                    "tapered_summary": tap_sum,
                    "comparison": comp,
                    "weak": weak,
                }
            )

    np.savez(
        OUTPUT_DIR / "catmap_variant_convergence.npz",
        n_values=np.array(n_values, dtype=int),
        order_values=np.array(order_values, dtype=int),
        l1_mat=l1_mat,
        l2_mat=l2_mat,
        weak_fourier_max_mat=weak_fourier_max_mat,
        weak_fourier_mean_mat=weak_fourier_mean_mat,
        baseline_flatness_mat=baseline_flatness_mat,
        tapered_flatness_mat=tapered_flatness_mat,
        baseline_entropy_mat=baseline_entropy_mat,
        tapered_entropy_mat=tapered_entropy_mat,
        baseline_tv_mat=baseline_tv_mat,
        tapered_tv_mat=tapered_tv_mat,
        baseline_peak_mass_mat=baseline_peak_mass_mat,
        tapered_peak_mass_mat=tapered_peak_mass_mat,
        baseline_peak_height_mat=baseline_peak_height_mat,
        tapered_peak_height_mat=tapered_peak_height_mat,
        records=np.array(records, dtype=object),
    )

    figs = []

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            l1_mat,
            title="Cat map: L1 distance (baseline vs tapered)",
            save_path=PLOT_DIR / "catmap_variant_l1.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            l2_mat,
            title="Cat map: L2 distance (baseline vs tapered)",
            save_path=PLOT_DIR / "catmap_variant_l2.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            weak_fourier_max_mat,
            title="Cat map: weak Fourier max discrepancy",
            save_path=PLOT_DIR / "catmap_variant_weak_fourier_max.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            weak_fourier_mean_mat,
            title="Cat map: weak Fourier mean discrepancy",
            save_path=PLOT_DIR / "catmap_variant_weak_fourier_mean.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            baseline_flatness_mat,
            title="Cat map baseline flatness",
            save_path=PLOT_DIR / "catmap_baseline_flatness.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            tapered_flatness_mat,
            title="Cat map tapered flatness",
            save_path=PLOT_DIR / "catmap_tapered_flatness.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            baseline_tv_mat,
            title="Cat map baseline total variation",
            save_path=PLOT_DIR / "catmap_baseline_total_variation.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            tapered_tv_mat,
            title="Cat map tapered total variation",
            save_path=PLOT_DIR / "catmap_tapered_total_variation.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            baseline_peak_mass_mat,
            title="Cat map baseline peak-mass ratio",
            save_path=PLOT_DIR / "catmap_baseline_peak_mass.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            tapered_peak_mass_mat,
            title="Cat map tapered peak-mass ratio",
            save_path=PLOT_DIR / "catmap_tapered_peak_mass.png",
        )[0]
    )

    # Slice plots versus n for fixed orders
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, order in enumerate(order_values):
        ax.plot(n_values, l1_mat[:, j], marker="o", label=f"order={order}")
    ax.set_xlabel("Trajectory length n")
    ax.set_ylabel("L1 distance")
    ax.set_title("Cat map: L1 distance vs n")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "catmap_variant_l1_vs_n.png", dpi=160)
    figs.append(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for j, order in enumerate(order_values):
        ax2.plot(n_values, weak_fourier_max_mat[:, j], marker="o", label=f"order={order}")
    ax2.set_xlabel("Trajectory length n")
    ax2.set_ylabel("Weak Fourier max discrepancy")
    ax2.set_title("Cat map: weak Fourier discrepancy vs n")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "catmap_variant_weak_vs_n.png", dpi=160)
    figs.append(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for j, order in enumerate(order_values):
        ax3.plot(n_values, baseline_tv_mat[:, j], marker="o", label=f"baseline, order={order}")
        ax3.plot(n_values, tapered_tv_mat[:, j], marker="x", linestyle="--", label=f"tapered, order={order}")
    ax3.set_xlabel("Trajectory length n")
    ax3.set_ylabel("Total variation")
    ax3.set_title("Cat map: total variation vs n")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(PLOT_DIR / "catmap_variant_total_variation_vs_n.png", dpi=160)
    figs.append(fig3)

    plt.show()


if __name__ == "__main__":
    main()
