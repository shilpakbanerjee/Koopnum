from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.cd_kernel.variants.cd_kernel_v001_baseline import fit_cd_kernel_baseline


OUTPUT_DIR = Path("experiments/cd_kernel/outputs/rotation")
PLOT_DIR = Path("experiments/cd_kernel/plots/rotation")


def generate_rotation(n: int = 2000, theta: float = 0.2) -> np.ndarray:
    X = np.zeros((n, 2), dtype=float)
    X[0] = [1.0, 0.0]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for k in range(n - 1):
        X[k + 1] = R @ X[k]
    return X


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    X = generate_rotation(n=2500, theta=0.17)
    result = fit_cd_kernel_baseline(X, order=80, grid_size=2048, regularization=1e-6)
    print(result.summary())

    result.save_npz(OUTPUT_DIR / "rotation_cd_result.npz")
    result.save_metadata_json(OUTPUT_DIR / "rotation_cd_metadata.json")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    result.plot_kernel(ax=axes[0])
    result.plot_density(ax=axes[1])
    fig.savefig(PLOT_DIR / "rotation_cd_baseline.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
