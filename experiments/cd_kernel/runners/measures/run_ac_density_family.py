"""
Generic runner for absolutely continuous measures on the unit circle.

This script allows testing the CD-kernel reconstruction pipeline on a
family of user-selectable absolutely continuous densities on [0, 2π).

Available example densities:
- uniform
- cosine
- shifted_cosine
- double_bump
- wrapped_gaussian

Outputs:
- saved numerical reconstruction data
- density comparison plots
- error plots
- kernel diagonal plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict
import numpy as np
import matplotlib.pyplot as plt

from experiments.cd_kernel.measures.benchmark_measures import (
    AbsolutelyContinuousMeasure,
    uniform_density,
    cosine_density,
    wrapped_gaussian_density,
)
from experiments.cd_kernel.measures.reconstruction import evaluate_cd_kernel_from_moments
from experiments.cd_kernel.runners.common_plotting import (
    save_true_vs_reconstructed_density_plot,
    save_kernel_comparison_plot,
    save_error_plot,
)


Array = np.ndarray


def circle_integral(values: Array, angles: Array) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.real(np.trapezoid(values, angles)))
    return float(np.real(np.trapz(values, angles)))


def normalize_density_values(values: Array, angles: Array) -> Array:
    values = np.asarray(values, dtype=float)
    integral = circle_integral(values, angles)
    if integral <= 0:
        raise ValueError("Density integral must be positive")
    return values / integral


def shifted_cosine_density(alpha: float = 0.5, shift: float = 0.0) -> Callable[[Array], Array]:
    if abs(alpha) >= 1.0:
        raise ValueError("Need |alpha| < 1 for nonnegative density")

    def rho(angles: Array) -> Array:
        return (1.0 + alpha * np.cos(angles - shift)) / (2.0 * np.pi)

    return rho


def double_bump_density(
    center1: float = 1.0,
    center2: float = 4.0,
    sigma1: float = 0.30,
    sigma2: float = 0.45,
    weight1: float = 0.55,
    weight2: float = 0.45,
) -> Callable[[Array], Array]:
    if weight1 < 0 or weight2 < 0:
        raise ValueError("Weights must be nonnegative")
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("Sigmas must be positive")

    g1 = wrapped_gaussian_density(center1, sigma1)
    g2 = wrapped_gaussian_density(center2, sigma2)

    def rho(angles: Array) -> Array:
        return weight1 * g1(angles) + weight2 * g2(angles)

    return rho


def get_density_registry() -> Dict[str, Callable[[], Callable[[Array], Array]]]:
    return {
        "uniform": lambda: uniform_density,
        "cosine": lambda: cosine_density(alpha=0.6),
        "shifted_cosine": lambda: shifted_cosine_density(alpha=0.75, shift=1.1),
        "double_bump": lambda: double_bump_density(
            center1=1.0,
            center2=4.2,
            sigma1=0.22,
            sigma2=0.40,
            weight1=0.6,
            weight2=0.4,
        ),
        "wrapped_gaussian": lambda: wrapped_gaussian_density(center=2.0, sigma=0.35),
    }


def run_density_case(
    density_name: str,
    density_fn: Callable[[Array], Array],
    order: int = 60,
    reconstruction_grid_size: int = 2048,
    moment_grid_size: int = 4096,
    regularization: float = 1e-8,
    show_plot: bool = True,
) -> None:
    output_dir = Path(f"experiments/cd_kernel/outputs/measures/{density_name}")
    plot_dir = Path(f"experiments/cd_kernel/plots/measures/{density_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    mu = AbsolutelyContinuousMeasure(density_fn)
    moments = mu.moments(
        order=order,
        grid_size=moment_grid_size,
        normalize=True,
    )

    result = evaluate_cd_kernel_from_moments(
        moments,
        order=order,
        grid_size=reconstruction_grid_size,
        regularization=regularization,
        normalize_density=True,
    )

    true_density = np.asarray(density_fn(result.angles), dtype=float)
    true_density = normalize_density_values(true_density, result.angles)

    print(f"\n=== Absolutely continuous density: {density_name} ===")
    print("Toeplitz condition number:", result.metadata["toeplitz_condition_number"])
    print("Top peaks / local maxima:")
    for item in result.top_peaks(k=8, min_separation=12):
        print(
            f"  angle={item['angle']:.6f}, "
            f"value={item['value']:.6e}, "
            f"point={item['point']}"
        )

    np.savez(
        output_dir / f"{density_name}_results.npz",
        moments=result.moments,
        toeplitz=result.toeplitz,
        angles=result.angles,
        circle_points=result.circle_points,
        kernel_diag=result.kernel_diag,
        density_proxy=result.density_proxy,
        true_density=true_density,
    )

    fig1, _ = save_true_vs_reconstructed_density_plot(
        angles=result.angles,
        true_density=true_density,
        reconstructed_density=result.density_proxy,
        title=f"Density comparison: {density_name}",
        save_path=plot_dir / f"{density_name}_density.png",
    )

    fig2, _ = save_kernel_comparison_plot(
        results=[result],
        labels=["Kernel diagonal"],
        title=f"Kernel diagonal: {density_name}",
        save_path=plot_dir / f"{density_name}_kernel_diag.png",
    )

    fig3, _ = save_error_plot(
        angles=result.angles,
        true_density=true_density,
        reconstructed_density=result.density_proxy,
        title=f"Reconstruction error: {density_name}",
        save_path=plot_dir / f"{density_name}_error.png",
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


def main():
    density_registry = get_density_registry()

    for density_name, density_factory in density_registry.items():
        density_fn = density_factory()
        run_density_case(
            density_name=density_name,
            density_fn=density_fn,
            order=60,
            reconstruction_grid_size=2048,
            moment_grid_size=4096,
            regularization=1e-8,
            show_plot=False,
        )

    print("\nFinished running all absolutely continuous density tests.")


if __name__ == "__main__":
    main()