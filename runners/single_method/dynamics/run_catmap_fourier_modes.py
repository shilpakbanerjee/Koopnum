"""
Compare CD-kernel reconstructions for several torus Fourier observables
on the Arnold cat map.

This runner uses the shared spectral-measure pipeline and compares
baseline and tapered reconstructions for several torus Fourier modes.

Fourier modes are the natural observables on the torus and usually give
cleaner spectral diagnostics than generic linear observables in the
ambient coordinates.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from methods.common.systems import generate_cat_map
from methods.common.observables import torus_fourier_mode
from methods.cd_kernel.dynamics.spectral_measure import (
    reconstruct_spectral_measure_from_system,
)
from methods.cd_kernel.core.cd_kernel_v002_tapered import (
    fit_cd_kernel_tapered_from_moments,
)
from methods.common.plotting.common_plotting import (
    save_density_comparison_plot,
    save_density_comparison_normalized_plot,
    save_density_comparison_log_plot,
    save_difference_plot,
)


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/dynamics/catmap_fourier_modes")
PLOT_DIR = Path("experiments/cd_kernel/plots/dynamics/catmap_fourier_modes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    n = 8000
    order = 100
    grid_size = 2048
    regularization = 1e-6

    modes = [
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 1),
    ]

    X = None
    baseline_results = {}
    tapered_results = {}
    spectral_data = {}

    for k1, k2 in modes:
        obs = torus_fourier_mode(k1, k2)

        X_mode, spec, baseline = reconstruct_spectral_measure_from_system(
            system_fn=generate_cat_map,
            system_kwargs={"n": n},
            order=order,
            observable=obs,
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
        tapered.metadata["source"] = "trajectory"
        tapered.metadata["system_name"] = "generate_cat_map"
        tapered.metadata["system_kwargs"] = {"n": n}
        tapered.metadata["observable_name"] = getattr(obs, "__name__", str(obs))
        tapered.metadata["signal_length"] = len(spec.signal)

        if X is None:
            X = X_mode

        baseline_results[(k1, k2)] = baseline
        tapered_results[(k1, k2)] = tapered
        spectral_data[(k1, k2)] = spec

        print(f"\n=== Cat map, mode ({k1},{k2}) ===")
        print("Baseline top peaks:")
        for item in baseline.top_peaks(k=8, min_separation=12):
            print(f"  angle={item['angle']:.6f}, value={item['value']:.6e}")

        print("Tapered top peaks:")
        for item in tapered.top_peaks(k=8, min_separation=12):
            print(f"  angle={item['angle']:.6f}, value={item['value']:.6e}")

        np.savez(
            OUTPUT_DIR / f"catmap_mode_{k1}_{k2}.npz",
            signal=spec.signal,
            moments=spec.moments,
            baseline_angles=baseline.angles,
            baseline_density=baseline.density_proxy,
            baseline_kernel=baseline.kernel_diag,
            tapered_angles=tapered.angles,
            tapered_density=tapered.density_proxy,
            tapered_kernel=tapered.kernel_diag,
        )

    # -----------------------------------------
    # Overlay baseline densities for all modes
    # -----------------------------------------
    baseline_list = [baseline_results[m] for m in modes]
    baseline_labels = [f"({k1},{k2})" for (k1, k2) in modes]

    save_density_comparison_plot(
        results=baseline_list,
        labels=baseline_labels,
        title="Cat map: baseline CD densities for torus Fourier modes",
        save_path=PLOT_DIR / "catmap_fourier_modes_baseline.png",
    )

    save_density_comparison_log_plot(
        results=baseline_list,
        labels=baseline_labels,
        title="Cat map: baseline densities for torus Fourier modes (log scale)",
        save_path=PLOT_DIR / "catmap_fourier_modes_baseline_log.png",
    )

    # -----------------------------------------
    # Overlay tapered densities for all modes
    # -----------------------------------------
    tapered_list = [tapered_results[m] for m in modes]
    tapered_labels = [f"({k1},{k2})" for (k1, k2) in modes]

    save_density_comparison_plot(
        results=tapered_list,
        labels=tapered_labels,
        title="Cat map: tapered CD densities for torus Fourier modes",
        save_path=PLOT_DIR / "catmap_fourier_modes_tapered.png",
    )

    save_density_comparison_log_plot(
        results=tapered_list,
        labels=tapered_labels,
        title="Cat map: tapered densities for torus Fourier modes (log scale)",
        save_path=PLOT_DIR / "catmap_fourier_modes_tapered_log.png",
    )

    # -----------------------------------------
    # Max-normalized tapered densities
    # -----------------------------------------
    save_density_comparison_normalized_plot(
        results=tapered_list,
        labels=tapered_labels,
        title="Cat map: tapered max-normalized densities by Fourier mode",
        save_path=PLOT_DIR / "catmap_fourier_modes_tapered_normalized.png",
    )

    # -----------------------------------------
    # Per-mode baseline vs tapered comparison
    # -----------------------------------------
    fig, axes = plt.subplots(len(modes), 1, figsize=(10, 3.2 * len(modes)), sharex=True)
    if len(modes) == 1:
        axes = [axes]

    for ax, (k1, k2) in zip(axes, modes):
        b = baseline_results[(k1, k2)]
        t = tapered_results[(k1, k2)]

        ax.plot(b.angles, b.density_proxy, lw=1.2, alpha=0.85, label="Baseline")
        ax.plot(t.angles, t.density_proxy, lw=1.2, alpha=0.90, linestyle="--", label="Tapered")
        ax.set_ylabel(f"Mode ({k1},{k2})")
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    axes[-1].set_xlabel("Angle on unit circle")
    fig.suptitle("Cat map: baseline vs tapered by torus Fourier mode", y=0.995)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "catmap_fourier_modes_compare.png", dpi=160)

    # -----------------------------------------
    # Per-mode absolute differences
    # -----------------------------------------
    for k1, k2 in modes:
        save_difference_plot(
            result_a=baseline_results[(k1, k2)],
            result_b=tapered_results[(k1, k2)],
            title=f"Cat map mode ({k1},{k2}): |baseline - tapered|",
            save_path=PLOT_DIR / f"catmap_mode_{k1}_{k2}_difference.png",
        )

    plt.show()


if __name__ == "__main__":
    main()