"""
Convergence study for exact-vs-empirical spectral reconstruction
in the planar rotation case.

This runner studies how the empirical CD-kernel reconstruction converges
to the exact atomic spectral measure as:

- trajectory length n increases
- moment truncation order increases

For each (n, order), it computes:
- shape diagnostics
- weak-convergence diagnostics

and saves the results in a structured .npz bundle.

Mathematical purpose
--------------------
This experiment separates:
    (i) reconstruction error of the CD-kernel method
from
    (ii) moment-estimation error coming from finite trajectory data.

Since the spectral measure for the rotation + complex-coordinate observable
is known exactly, this is the cleanest first convergence benchmark.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.core.moment_sources import (
    ExactMomentSource,
    EmpiricalMomentSource,
)
from experiments.cd_kernel.core.kernel import evaluate_cd_kernel_from_moments
from experiments.cd_kernel.core.diagnostics import (
    summarize_result,
    compare_results,
    weak_convergence_summary,
)
from experiments.cd_kernel.dynamics.systems import generate_planar_rotation
from experiments.cd_kernel.dynamics.observables import complex_coordinate


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/convergence/rotation_exact_vs_empirical")
PLOT_DIR = Path("experiments/cd_kernel/plots/convergence/rotation_exact_vs_empirical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def atomic_moment_function(theta: float):
    def mk(k: int) -> complex:
        return np.exp(-1j * k * theta)
    return mk


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
    theta = 0.35
    grid_size = 2048
    regularization = 1e-6

    # Convergence grids
    n_values = [250, 500, 1000, 2000, 4000]
    order_values = [10, 20, 40, 60, 80]

    observable = complex_coordinate(0, 1)

    # Store metric matrices indexed by [i_n, i_order]
    l1_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    l2_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    weak_fourier_max_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    weak_fourier_mean_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    flatness_emp_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    entropy_emp_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    peak_mass_emp_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    exact_peak_height_mat = np.zeros((len(n_values), len(order_values)), dtype=float)
    emp_peak_height_mat = np.zeros((len(n_values), len(order_values)), dtype=float)

    # Optional detailed storage
    records = []

    for i_n, n in enumerate(n_values):
        print(f"\n=== n = {n} ===")

        X = generate_planar_rotation(n=n, theta=theta)
        signal = observable(X)

        for i_o, order in enumerate(order_values):
            print(f"  order = {order}")

            # Exact source
            exact_source = ExactMomentSource(atomic_moment_function(theta))
            exact_moments = exact_source.moments(order)

            exact_result = evaluate_cd_kernel_from_moments(
                exact_moments,
                order=order,
                grid_size=grid_size,
                regularization=regularization,
                normalize_density=True,
            )

            # Empirical source
            empirical_source = EmpiricalMomentSource(signal)
            empirical_moments = empirical_source.moments(order)

            empirical_result = evaluate_cd_kernel_from_moments(
                empirical_moments,
                order=order,
                grid_size=grid_size,
                regularization=regularization,
                normalize_density=True,
            )

            comp = compare_results(exact_result, empirical_result)
            weak = weak_convergence_summary(exact_result, empirical_result, max_mode=12)
            exact_sum = summarize_result(exact_result)
            emp_sum = summarize_result(empirical_result)

            l1_mat[i_n, i_o] = comp["l1_distance"]
            l2_mat[i_n, i_o] = comp["l2_distance"]

            weak_fourier_max_mat[i_n, i_o] = weak["fourier_max_abs_discrepancy"]
            weak_fourier_mean_mat[i_n, i_o] = weak["fourier_mean_abs_discrepancy"]

            flatness_emp_mat[i_n, i_o] = emp_sum["spectral_flatness"]
            entropy_emp_mat[i_n, i_o] = emp_sum["entropy"]
            peak_mass_emp_mat[i_n, i_o] = emp_sum["peak_mass_ratio"]

            exact_peak_height_mat[i_n, i_o] = exact_sum["max_peak_height"]
            emp_peak_height_mat[i_n, i_o] = emp_sum["max_peak_height"]

            records.append(
                {
                    "n": n,
                    "order": order,
                    "comparison": comp,
                    "weak": weak,
                    "exact_summary": exact_sum,
                    "empirical_summary": emp_sum,
                }
            )

    # Save raw numeric outputs
    np.savez(
        OUTPUT_DIR / "rotation_exact_vs_empirical_convergence.npz",
        theta=theta,
        n_values=np.array(n_values, dtype=int),
        order_values=np.array(order_values, dtype=int),
        l1_mat=l1_mat,
        l2_mat=l2_mat,
        weak_fourier_max_mat=weak_fourier_max_mat,
        weak_fourier_mean_mat=weak_fourier_mean_mat,
        flatness_emp_mat=flatness_emp_mat,
        entropy_emp_mat=entropy_emp_mat,
        peak_mass_emp_mat=peak_mass_emp_mat,
        exact_peak_height_mat=exact_peak_height_mat,
        emp_peak_height_mat=emp_peak_height_mat,
        records=np.array(records, dtype=object),
    )

    # Heatmaps
    figs = []

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            l1_mat,
            title="Rotation convergence: L1 distance (exact vs empirical)",
            save_path=PLOT_DIR / "rotation_convergence_l1.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            l2_mat,
            title="Rotation convergence: L2 distance (exact vs empirical)",
            save_path=PLOT_DIR / "rotation_convergence_l2.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            weak_fourier_max_mat,
            title="Rotation convergence: weak Fourier max discrepancy",
            save_path=PLOT_DIR / "rotation_convergence_weak_fourier_max.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            weak_fourier_mean_mat,
            title="Rotation convergence: weak Fourier mean discrepancy",
            save_path=PLOT_DIR / "rotation_convergence_weak_fourier_mean.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            flatness_emp_mat,
            title="Rotation empirical flatness",
            save_path=PLOT_DIR / "rotation_convergence_flatness.png",
        )[0]
    )

    figs.append(
        metric_matrix_to_heatmap(
            n_values,
            order_values,
            peak_mass_emp_mat,
            title="Rotation empirical peak-mass ratio",
            save_path=PLOT_DIR / "rotation_convergence_peak_mass.png",
        )[0]
    )

    # Slice plots vs n for fixed orders
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, order in enumerate(order_values):
        ax.plot(n_values, l1_mat[:, j], marker="o", label=f"order={order}")
    ax.set_xlabel("Trajectory length n")
    ax.set_ylabel("L1 distance")
    ax.set_title("Rotation convergence: L1 distance vs n")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rotation_convergence_l1_vs_n.png", dpi=160)
    figs.append(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for j, order in enumerate(order_values):
        ax2.plot(n_values, weak_fourier_max_mat[:, j], marker="o", label=f"order={order}")
    ax2.set_xlabel("Trajectory length n")
    ax2.set_ylabel("Weak Fourier max discrepancy")
    ax2.set_title("Rotation convergence: weak Fourier discrepancy vs n")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "rotation_convergence_weak_vs_n.png", dpi=160)
    figs.append(fig2)

    plt.show()


if __name__ == "__main__":
    main()