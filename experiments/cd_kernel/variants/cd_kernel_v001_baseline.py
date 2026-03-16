from __future__ import annotations

"""
Baseline Christoffel--Darboux kernel spectral estimator for Koopman analysis.

This file is intentionally experiment-facing rather than library-facing. The goal is
for you to modify the algorithm here while keeping the core package stable.

Main references
---------------
1. Milan Korda, Mihai Putinar, Igor Mezić,
   "Data-driven spectral analysis of the Koopman operator,"
   Applied and Computational Harmonic Analysis 48(2), 599--629, 2020.
   DOI: 10.1016/j.acha.2018.06.008

2. Hassan Arbabi, Igor Mezić,
   "Ergodic theory, dynamic mode decomposition, and computation of spectral
   properties of the Koopman operator,"
   SIAM Journal on Applied Dynamical Systems 16(4), 2096--2126, 2017.
   DOI: 10.1137/17M1125236

What this baseline does
-----------------------
- estimates the moment sequence of the observable's spectral measure from one long
  trajectory;
- builds a Toeplitz moment matrix;
- evaluates a Christoffel-function / CD-kernel proxy on the unit circle;
- returns simple diagnostics for point-mass detection and density visualization.

What it does NOT yet do
-----------------------
- full OPUC recurrence construction;
- atomic mass estimation with sharp asymptotics;
- rigorous decomposition into pure point / a.c. / singular continuous parts.

Still, this is a solid baseline for experiments and aligns with the core workflow in
Korda--Putinar--Mezić: moments from trajectories, Toeplitz structure, and kernel-based
spectral diagnostics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

Array = np.ndarray
Observable = Callable[[Array], Union[complex, float]]

@dataclass(frozen=True)
class Reference:
    authors: str
    title: str
    venue: str
    year: int
    doi_or_url: str | None = None


REFERENCES = [
    Reference(
        authors="Milan Korda, Mihai Putinar, Igor Mezić",
        title="Data-driven spectral analysis of the Koopman operator",
        venue="Applied and Computational Harmonic Analysis 48(2), 599-629",
        year=2020,
        doi_or_url="10.1016/j.acha.2018.06.008",
    ),
    Reference(
        authors="Hassan Arbabi, Igor Mezić",
        title="Ergodic theory, dynamic mode decomposition, and computation of spectral properties of the Koopman operator",
        venue="SIAM Journal on Applied Dynamical Systems 16(4), 2096-2126",
        year=2017,
        doi_or_url="10.1137/17M1125236",
    ),
]


@dataclass
class CDKernelResult:
    observable_values: Array
    moments: Array
    toeplitz_matrix: Array
    support_angles: Array
    support_points: Array
    christoffel_function: Array
    kernel_diagonal: Array
    density_proxy: Array
    metadata: dict = field(default_factory=dict)

    def top_atoms(self, k: int = 10, min_separation: int = 3) -> list[tuple[float, float]]:
        """Return the strongest peaks as (angle, score)."""
        scores = np.asarray(self.kernel_diagonal, dtype=float)
        order = np.argsort(scores)[::-1]
        selected: list[int] = []
        for idx in order:
            if all(abs(idx - j) > min_separation for j in selected):
                selected.append(int(idx))
            if len(selected) >= k:
                break
        return [(float(self.support_angles[i]), float(scores[i])) for i in selected]

    def summary(self) -> str:
        lines = [
            "CD-kernel baseline summary",
            f"  order                : {self.metadata.get('order')}",
            f"  grid_size            : {self.metadata.get('grid_size')}",
            f"  regularization       : {self.metadata.get('regularization')}",
            f"  taper                : {self.metadata.get('taper')}",
            f"  top peaks (angle, K) : {self.top_atoms(k=5)}",
        ]
        return "\n".join(lines)

    def plot_density(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.support_angles, self.density_proxy)
        ax.set_xlabel("angle")
        ax.set_ylabel("density proxy")
        ax.set_title("CD-kernel density proxy")
        return ax

    def plot_kernel(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.support_angles, self.kernel_diagonal)
        ax.set_xlabel("angle")
        ax.set_ylabel("K_N(z,z)")
        ax.set_title("CD-kernel diagonal")
        return ax

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            observable_values=self.observable_values,
            moments=self.moments,
            toeplitz_matrix=self.toeplitz_matrix,
            support_angles=self.support_angles,
            support_points=self.support_points,
            christoffel_function=self.christoffel_function,
            kernel_diagonal=self.kernel_diagonal,
            density_proxy=self.density_proxy,
        )

    def save_metadata_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _as_trajectory(X: Array) -> Array:
    X = np.asarray(X)
    if X.ndim == 1:
        return X[:, None]
    if X.ndim != 2:
        raise ValueError("Trajectory must have shape (n_samples, n_features) or (n_samples,).")
    return X



def _default_observable(x: Array) -> complex:
    x = np.asarray(x).reshape(-1)
    if x.size == 1:
        return complex(x[0])
    return complex(x[0] + 1j * x[1])



def evaluate_observable(X: Array, observable: Optional[Observable] = None) -> Array:
    X = _as_trajectory(X)
    if observable is None:
        observable = _default_observable
    values = np.asarray([observable(x) for x in X], dtype=np.complex128)
    return values



def estimate_moments(
    observable_values: Array,
    order: int,
    center: bool = True,
    normalize: bool = True,
    taper: str = "none",
) -> Array:
    """
    Estimate moments m_k = <U^k f, f> / <f, f> from one trajectory.

    Parameters
    ----------
    observable_values:
        Complex time series f(x_0), ..., f(x_{N-1}).
    order:
        Maximum lag to compute.
    center:
        Subtract empirical mean first.
    normalize:
        Normalize by m_0.
    taper:
        One of {'none', 'bartlett', 'hann'} applied to the lag sequence to stabilize
        high-order moment estimation.
    """
    y = np.asarray(observable_values, dtype=np.complex128).reshape(-1)
    if y.size <= order:
        raise ValueError("Need number of samples > order.")
    if center:
        y = y - np.mean(y)

    n = y.size
    moments = np.empty(order + 1, dtype=np.complex128)
    for k in range(order + 1):
        moments[k] = np.vdot(y[: n - k], y[k:]) / (n - k)

    if taper == "bartlett":
        weights = 1.0 - np.arange(order + 1) / (order + 1)
        moments *= weights
    elif taper == "hann":
        weights = 0.5 * (1.0 + np.cos(np.pi * np.arange(order + 1) / (order + 1)))
        moments *= weights
    elif taper != "none":
        raise ValueError("Unknown taper. Use 'none', 'bartlett', or 'hann'.")

    if normalize:
        if abs(moments[0]) < 1e-14:
            raise ValueError("Zeroth moment is numerically zero; cannot normalize.")
        moments = moments / moments[0]
    return moments



def toeplitz_from_moments(moments: Array, hermitian: bool = True) -> Array:
    """
    Build T_N = [m_{j-k}] using conjugate symmetry m_{-k} = conjugate(m_k).
    """
    moments = np.asarray(moments, dtype=np.complex128)
    order = len(moments) - 1
    T = np.empty((order + 1, order + 1), dtype=np.complex128)
    for j in range(order + 1):
        for k in range(order + 1):
            lag = j - k
            if lag >= 0:
                T[j, k] = moments[lag]
            else:
                T[j, k] = np.conjugate(moments[-lag]) if hermitian else moments[-lag]
    return T



def vandermonde_on_unit_circle(order: int, grid_size: int) -> tuple[Array, Array, Array]:
    angles = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
    z = np.exp(1j * angles)
    V = np.vstack([z**k for k in range(order + 1)])
    return angles, z, V



def evaluate_cd_kernel_from_toeplitz(
    moments: Array,
    grid_size: int = 2048,
    regularization: float = 1e-8,
    density_floor: float = 1e-14,
) -> tuple[Array, Array, Array, Array, Array]:
    """
    Evaluate the diagonal CD-kernel proxy K_N(z,z) = v(z)^* T_N^{-1} v(z).

    The reciprocal is used as a Christoffel function / density proxy.
    """
    order = len(moments) - 1
    T = toeplitz_from_moments(moments)
    T_reg = T + regularization * np.eye(order + 1)
    angles, z, V = vandermonde_on_unit_circle(order, grid_size)
    Tinv = np.linalg.inv(T_reg)

    # Efficient diagonal evaluation of V^* T^{-1} V
    TV = Tinv @ V
    kernel_diag = np.real(np.sum(np.conjugate(V) * TV, axis=0))
    kernel_diag = np.maximum(kernel_diag, density_floor)
    christoffel = 1.0 / kernel_diag

    # Normalize the proxy into a density-like curve on [0, 2π)
    integral = np.trapezoid(christoffel, angles)
    density_proxy = christoffel / integral if integral > 0 else christoffel
    return T, angles, z, kernel_diag, density_proxy


# -----------------------------------------------------------------------------
# Main experiment-facing API
# -----------------------------------------------------------------------------


def fit_cd_kernel_baseline(
    X: Array,
    observable: Optional[Observable] = None,
    order: int = 64,
    grid_size: int = 2048,
    regularization: float = 1e-8,
    center: bool = True,
    normalize: bool = True,
    taper: str = "none",
) -> CDKernelResult:
    """
    Fit the baseline CD-kernel spectral estimator on a trajectory.

    Parameters
    ----------
    X:
        Trajectory with shape (n_samples, n_features) or (n_samples,).
    observable:
        Scalar observable f. If omitted, uses x[0] + i x[1] for 2D and x[0] for 1D.
    order:
        Degree/order N of the moment matrix T_N.
    grid_size:
        Number of points on the unit circle for evaluation.
    regularization:
        Tikhonov regularization added to the Toeplitz matrix.
    center, normalize, taper:
        Moment-estimation controls.
    """
    values = evaluate_observable(X, observable=observable)
    moments = estimate_moments(
        values,
        order=order,
        center=center,
        normalize=normalize,
        taper=taper,
    )
    T, angles, z, kernel_diag, density_proxy = evaluate_cd_kernel_from_toeplitz(
        moments,
        grid_size=grid_size,
        regularization=regularization,
    )
    return CDKernelResult(
        observable_values=values,
        moments=moments,
        toeplitz_matrix=T,
        support_angles=angles,
        support_points=z,
        christoffel_function=density_proxy,
        kernel_diagonal=kernel_diag,
        density_proxy=density_proxy,
        metadata={
            "algorithm": "cd_kernel_v001_baseline",
            "order": order,
            "grid_size": grid_size,
            "regularization": regularization,
            "center": center,
            "normalize": normalize,
            "taper": taper,
            "references": [ref.__dict__ for ref in REFERENCES],
        },
    )


# -----------------------------------------------------------------------------
# Small local demo for quick checks
# -----------------------------------------------------------------------------


def _generate_rotation(n: int = 1000, theta: float = 0.15) -> Array:
    X = np.zeros((n, 2), dtype=float)
    X[0] = [1.0, 0.0]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for k in range(n - 1):
        X[k + 1] = R @ X[k]
    return X


if __name__ == "__main__":
    X = _generate_rotation(n=1500, theta=0.21)
    result = fit_cd_kernel_baseline(X, order=64, grid_size=2048, regularization=1e-6)
    print(result.summary())
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    result.plot_kernel(ax=axes[0])
    result.plot_density(ax=axes[1])
    plt.show()
