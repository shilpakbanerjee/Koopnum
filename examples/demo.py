from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from koopman_toolkit import (
    christoffel_darboux_spectral_estimation,
    default_generator,
    default_polynomial_edmd,
    dmd,
    hankel_dmd,
    moment_based_spectral_reconstruction,
)


def make_rotation(n: int = 600, theta: float = 0.12) -> np.ndarray:
    x = np.zeros((n, 2))
    x[0] = [1.0, 0.0]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for k in range(n - 1):
        x[k + 1] = R @ x[k]
    return x


if __name__ == "__main__":
    X = make_rotation()

    dmd_res = dmd(X, rank=2)
    edmd_res = default_polynomial_edmd(X, degree=2)
    hankel_res = hankel_dmd(X[:, 0], delays=25, rank=8)
    gen_res = default_generator(X, dt=1.0, degree=2)
    spec_res = moment_based_spectral_reconstruction(X[:, 0], max_lag=80)
    cd_res = christoffel_darboux_spectral_estimation(X[:, 0], order=40)

    print("DMD eigenvalues:\n", dmd_res.eigenvalues)
    print("EDMD eigenvalues (first 8):\n", edmd_res.eigenvalues[:8])
    print("Hankel DMD eigenvalues (first 8):\n", hankel_res.eigenvalues[:8])
    print("Generator eigenvalues (first 8):\n", gen_res.eigenvalues[:8])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(spec_res.support_angles, spec_res.density, label="Moment-based")
    ax.plot(cd_res.support_angles, cd_res.density, label="Christoffel-Darboux", alpha=0.8)
    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Density proxy")
    ax.legend()
    ax.set_title("Spectral estimates for a planar rotation")
    plt.tight_layout()
    plt.show()
