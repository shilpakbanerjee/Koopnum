from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from scipy.linalg import eigh, eig, lstsq, solve, svd

from .references import Reference, algorithm_metadata

Array = np.ndarray
FeatureMap = Callable[[Array], Array]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _as_2d_samples(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X[:, None]
    if X.ndim != 2:
        raise ValueError("Input must be 1D or 2D with shape (n_samples, n_features).")
    return X



def _pair_snapshots(X: Array) -> tuple[Array, Array]:
    X = _as_2d_samples(X)
    if X.shape[0] < 2:
        raise ValueError("At least two time samples are required.")
    return X[:-1], X[1:]



def _truncated_svd(X: Array, rank: Optional[int] = None) -> tuple[Array, Array, Array]:
    U, s, Vh = svd(X, full_matrices=False)
    if rank is None:
        rank = len(s)
    rank = max(1, min(rank, len(s)))
    return U[:, :rank], s[:rank], Vh[:rank]



def _pseudo_inverse_from_svd(U: Array, s: Array, Vh: Array, rcond: float = 1e-12) -> Array:
    cutoff = rcond * np.max(s)
    inv_s = np.array([1.0 / val if val > cutoff else 0.0 for val in s])
    return (Vh.T * inv_s) @ U.T



def _rbf_kernel(X: Array, Y: Array, gamma: float) -> Array:
    X2 = np.sum(X * X, axis=1)[:, None]
    Y2 = np.sum(Y * Y, axis=1)[None, :]
    dist2 = np.maximum(X2 + Y2 - 2.0 * X @ Y.T, 0.0)
    return np.exp(-gamma * dist2)



def _finite_difference(X: Array, dt: float) -> Array:
    return np.gradient(X, dt, axis=0, edge_order=2)


# -----------------------------------------------------------------------------
# Feature maps
# -----------------------------------------------------------------------------


class IdentityFeatures:
    def __call__(self, X: Array) -> Array:
        X = _as_2d_samples(X)
        return X.T


class PolynomialFeatures:
    """Simple explicit polynomial lifting for small/medium problems."""

    def __init__(self, degree: int = 2, include_bias: bool = True):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.include_bias = include_bias

    def __call__(self, X: Array) -> Array:
        X = _as_2d_samples(X)
        n, d = X.shape
        cols: list[Array] = []
        if self.include_bias:
            cols.append(np.ones((n, 1)))
        cols.append(X)
        if self.degree >= 2:
            for deg in range(2, self.degree + 1):
                cols.extend(_monomials_of_degree(X, deg))
        return np.hstack(cols).T



def _monomials_of_degree(X: Array, degree: int) -> list[Array]:
    from itertools import combinations_with_replacement

    n, d = X.shape
    out: list[Array] = []
    for idx in combinations_with_replacement(range(d), degree):
        term = np.ones(n)
        for j in idx:
            term *= X[:, j]
        out.append(term[:, None])
    return out


class DelayEmbeddingFeatures:
    def __init__(self, delays: int):
        if delays < 1:
            raise ValueError("delays must be >= 1")
        self.delays = delays

    def __call__(self, X: Array) -> Array:
        X = _as_2d_samples(X)
        n, d = X.shape
        if n <= self.delays:
            raise ValueError("Need more samples than delays.")
        H = []
        for t in range(self.delays, n):
            H.append(X[t - self.delays : t + 1].reshape(-1))
        return np.asarray(H, dtype=float).T


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------


@dataclass
class LinearKoopmanResult:
    operator: Array
    eigenvalues: Array
    eigenvectors: Array | None = None
    modes: Array | None = None
    singular_values: Array | None = None
    feature_map: FeatureMap | None = None
    encoder: Callable[[Array], Array] | None = None
    decoder: Callable[[Array], Array] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SpectralEstimateResult:
    moments: Array
    support_angles: Array
    density: Array
    kernel_values: Array | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SINDyResult:
    coefficients: Array
    library_names: list[str]
    rhs: Callable[[Array], Array]
    metadata: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Algorithms
# -----------------------------------------------------------------------------


@algorithm_metadata(
    name="Dynamic Mode Decomposition",
    references=[
        Reference(
            authors="Peter J. Schmid",
            title="Dynamic mode decomposition of numerical and experimental data",
            venue="Journal of Fluid Mechanics 656, 5-28",
            year=2010,
            doi_or_url="10.1017/S0022112010001217",
        )
    ],
    notes="Classical snapshot-based approximation; best suited to point-spectrum-dominated behavior.",
)
def dmd(X: Array, rank: Optional[int] = None, rcond: float = 1e-12) -> LinearKoopmanResult:
    """
    Compute Dynamic Mode Decomposition.

    Parameters
    ----------
    X:
        Time-ordered snapshots with shape (n_samples, n_features).
    rank:
        Optional truncation rank.
    rcond:
        SVD cutoff used in the pseudoinverse.

    Returns
    -------
    LinearKoopmanResult

    Reference
    ---------
    Schmid (2010), Journal of Fluid Mechanics.
    """
    X0, X1 = _pair_snapshots(X)
    U, s, Vh = _truncated_svd(X0.T, rank=rank)
    X0_pinv = _pseudo_inverse_from_svd(U, s, Vh, rcond=rcond)
    A = X1.T @ X0_pinv
    A_tilde = U.T @ X1.T @ Vh.T @ np.diag(1.0 / s)
    eigvals, W = eig(A_tilde)
    modes = X1.T @ Vh.T @ np.diag(1.0 / s) @ W
    return LinearKoopmanResult(
        operator=A,
        eigenvalues=eigvals,
        eigenvectors=W,
        modes=modes,
        singular_values=s,
        feature_map=IdentityFeatures(),
        metadata={"rank": rank or len(s)},
    )


@algorithm_metadata(
    name="Koopman Mode Decomposition",
    references=[
        Reference(
            authors="Igor Mezić",
            title="Spectral Properties of Dynamical Systems, Model Reduction and Decompositions",
            venue="Nonlinear Dynamics 41, 309-325",
            year=2005,
            doi_or_url="10.1007/s11071-005-2824-x",
        ),
        Reference(
            authors="Peter J. Schmid",
            title="Dynamic mode decomposition of numerical and experimental data",
            venue="Journal of Fluid Mechanics 656, 5-28",
            year=2010,
            doi_or_url="10.1017/S0022112010001217",
        ),
    ],
    notes="Operationally extracted from DMD in the standard snapshot setting.",
)
def koopman_mode_decomposition(X: Array, rank: Optional[int] = None) -> LinearKoopmanResult:
    """Return DMD-based Koopman modes and eigenvalues."""
    result = dmd(X, rank=rank)
    result.metadata["interpretation"] = "DMD-based Koopman mode decomposition"
    return result


@algorithm_metadata(
    name="Extended Dynamic Mode Decomposition",
    references=[
        Reference(
            authors="Matthew O. Williams, Ioannis G. Kevrekidis, Clarence W. Rowley",
            title="A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition",
            venue="Journal of Nonlinear Science 25, 1307-1346",
            year=2015,
            doi_or_url="10.1007/s00332-015-9258-5",
        )
    ],
    notes="Galerkin projection in a chosen feature dictionary.",
)
def edmd(X: Array, feature_map: FeatureMap, reg: float = 1e-8) -> LinearKoopmanResult:
    """
    Compute EDMD in an explicit observable dictionary.

    The feature map should return an array of shape
    (n_lifted_features, n_samples).
    """
    X0, X1 = _pair_snapshots(X)
    Psi0 = np.asarray(feature_map(X0), dtype=float)
    Psi1 = np.asarray(feature_map(X1), dtype=float)
    G = (Psi0 @ Psi0.T) / Psi0.shape[1]
    A = (Psi0 @ Psi1.T) / Psi0.shape[1]
    K = solve(G + reg * np.eye(G.shape[0]), A, assume_a="pos")
    eigvals, eigvecs = eig(K)
    return LinearKoopmanResult(
        operator=K,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        feature_map=feature_map,
        metadata={"regularization": reg, "lifted_dimension": Psi0.shape[0]},
    )


@algorithm_metadata(
    name="Kernel EDMD",
    references=[
        Reference(
            authors="Matthew O. Williams, Clarence W. Rowley, Ioannis G. Kevrekidis",
            title="A Kernel-Based Method for Data-Driven Koopman Spectral Analysis",
            venue="Journal of Computational Dynamics 2(2), 247-265",
            year=2015,
            doi_or_url="10.3934/jcd.2015005",
        )
    ],
    notes="Implicit high-dimensional lifting through kernels.",
)
def kernel_edmd(X: Array, gamma: float = 1.0, reg: float = 1e-8) -> LinearKoopmanResult:
    """Compute a simple RBF-kernel EDMD approximation."""
    X0, X1 = _pair_snapshots(X)
    Kxx = _rbf_kernel(X0, X0, gamma)
    Kxy = _rbf_kernel(X0, X1, gamma)
    A = solve(Kxx + reg * np.eye(Kxx.shape[0]), Kxy, assume_a="pos")
    eigvals, eigvecs = eig(A)
    return LinearKoopmanResult(
        operator=A,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        metadata={"gamma": gamma, "regularization": reg, "n_snapshots": X0.shape[0]},
    )


@algorithm_metadata(
    name="Sparse Identification of Nonlinear Dynamics",
    references=[
        Reference(
            authors="Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz",
            title="Discovering governing equations from data by sparse identification of nonlinear dynamical systems",
            venue="PNAS 113(15), 3932-3937",
            year=2016,
            doi_or_url="10.1073/pnas.1517384113",
        )
    ],
    notes="Not a direct Koopman approximation, but often paired with Koopman-style reduced models.",
)
def sindy(
    X: Array,
    dt: float,
    degree: int = 2,
    threshold: float = 1e-2,
    max_iter: int = 10,
) -> SINDyResult:
    """Identify a sparse ODE model from trajectory data using STLSQ."""
    X = _as_2d_samples(X)
    dXdt = _finite_difference(X, dt)
    Theta, names = build_sindy_library(X, degree=degree)
    Xi, *_ = lstsq(Theta, dXdt)
    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0
        for col in range(Xi.shape[1]):
            keep = ~small[:, col]
            if np.any(keep):
                Xi[keep, col], *_ = lstsq(Theta[:, keep], dXdt[:, col])

    def rhs(x: Array) -> Array:
        x = _as_2d_samples(np.asarray(x, dtype=float))
        Theta_x, _ = build_sindy_library(x, degree=degree)
        return (Theta_x @ Xi).reshape(-1)

    return SINDyResult(
        coefficients=Xi,
        library_names=names,
        rhs=rhs,
        metadata={"dt": dt, "degree": degree, "threshold": threshold},
    )


@algorithm_metadata(
    name="Hankel DMD / Delay Embedding",
    references=[
        Reference(
            authors="Hassan Arbabi, Igor Mezić",
            title="Ergodic Theory, Dynamic Mode Decomposition, and Computation of Spectral Properties of the Koopman Operator",
            venue="SIAM Journal on Applied Dynamical Systems 16(4), 2096-2126",
            year=2017,
            doi_or_url="10.1137/17M1125236",
        )
    ],
    notes="Delay coordinates recover richer invariant subspaces and often improve spectral estimation.",
)
def hankel_dmd(X: Array, delays: int = 10, rank: Optional[int] = None) -> LinearKoopmanResult:
    """Apply DMD to delay-embedded snapshots."""
    X = _as_2d_samples(X)
    H = DelayEmbeddingFeatures(delays)(X).T
    result = dmd(H, rank=rank)
    result.metadata.update({"delays": delays, "embedded_dimension": H.shape[1]})
    return result


@algorithm_metadata(
    name="Generator-based Koopman Approximation",
    references=[
        Reference(
            authors="Dimitrios Giannakis",
            title="Data-driven spectral decomposition and forecasting of ergodic dynamical systems",
            venue="Applied and Computational Harmonic Analysis 47(2), 338-396",
            year=2019,
            doi_or_url="10.1016/j.acha.2017.09.001",
        )
    ],
    notes="Approximates the infinitesimal generator rather than the one-step Koopman operator.",
)
def koopman_generator(X: Array, dt: float, feature_map: FeatureMap, reg: float = 1e-8) -> LinearKoopmanResult:
    """Approximate the Koopman generator in a chosen feature basis."""
    X = _as_2d_samples(X)
    Phi = np.asarray(feature_map(X), dtype=float)
    dPhi = _finite_difference(Phi.T, dt).T
    G = (Phi @ Phi.T) / Phi.shape[1]
    A = (Phi @ dPhi.T) / Phi.shape[1]
    L = solve(G + reg * np.eye(G.shape[0]), A, assume_a="pos")
    eigvals, eigvecs = eig(L)
    return LinearKoopmanResult(
        operator=L,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        feature_map=feature_map,
        metadata={"dt": dt, "regularization": reg, "interpretation": "generator approximation"},
    )


@algorithm_metadata(
    name="Neural Koopman (lightweight linear-autoencoder surrogate)",
    references=[
        Reference(
            authors="Bethany Lusch, J. Nathan Kutz, Steven L. Brunton",
            title="Deep learning for universal linear embeddings of nonlinear dynamics",
            venue="Nature Communications 9, 4950",
            year=2018,
            doi_or_url="10.1038/s41467-018-07210-0",
        )
    ],
    notes="This implementation is a research-friendly PCA surrogate, not a full deep-learning stack.",
)
def neural_koopman_pca(X: Array, latent_dim: int = 2, reg: float = 1e-8) -> LinearKoopmanResult:
    """
    Lightweight surrogate for neural Koopman models.

    Uses PCA as encoder/decoder and learns a linear latent evolution.
    This keeps the API ready for replacement by a PyTorch/JAX model later.
    """
    X = _as_2d_samples(X)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, s, Vh = svd(Xc, full_matrices=False)
    encoder_matrix = Vh[:latent_dim].T

    def encoder(data: Array) -> Array:
        data = _as_2d_samples(data)
        return (data - mu) @ encoder_matrix

    def decoder(Z: Array) -> Array:
        Z = _as_2d_samples(Z)
        return Z @ encoder_matrix.T + mu

    Z = encoder(X)
    A, *_ = lstsq(Z[:-1], Z[1:])
    if reg > 0:
        A = solve(Z[:-1].T @ Z[:-1] + reg * np.eye(latent_dim), Z[:-1].T @ Z[1:])
    eigvals, eigvecs = eig(A)
    return LinearKoopmanResult(
        operator=A,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        encoder=encoder,
        decoder=decoder,
        metadata={"latent_dim": latent_dim, "singular_values": s[:latent_dim]},
    )


@algorithm_metadata(
    name="Moment-based Spectral Reconstruction",
    references=[
        Reference(
            authors="Milan Korda, Mihai Putinar, Igor Mezić",
            title="Data-driven spectral analysis of the Koopman operator",
            venue="Applied and Computational Harmonic Analysis 48(2), 599-629",
            year=2020,
            doi_or_url="10.1016/j.acha.2018.06.008",
        )
    ],
    notes="Uses autocorrelation moments of observables to reconstruct spectral information.",
)
def moment_based_spectral_reconstruction(y: Array, max_lag: int = 50, grid_size: int = 1024) -> SpectralEstimateResult:
    """
    Estimate a Fourier density proxy from empirical spectral moments.

    This is a practical callable approximation, not the full weak-convergence
    machinery of the paper.
    """
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    y = y - y.mean()
    n = len(y)
    moments = np.zeros(max_lag + 1, dtype=np.complex128)
    for k in range(max_lag + 1):
        moments[k] = np.vdot(y[: n - k], y[k:]) / (n - k)
    if np.abs(moments[0]) > 0:
        moments /= moments[0]
    theta = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    density = np.real(moments[0] + 2 * sum(moments[k] * np.exp(-1j * k * theta) for k in range(1, max_lag + 1)))
    density = np.maximum(density, 0.0)
    norm = np.trapezoid(density, theta)
    if norm > 0:
        density /= norm
    return SpectralEstimateResult(moments=moments, support_angles=theta, density=density, metadata={"max_lag": max_lag})


@algorithm_metadata(
    name="Christoffel-Darboux Spectral Estimation",
    references=[
        Reference(
            authors="Milan Korda, Mihai Putinar, Igor Mezić",
            title="Data-driven spectral analysis of the Koopman operator",
            venue="Applied and Computational Harmonic Analysis 48(2), 599-629",
            year=2020,
            doi_or_url="10.1016/j.acha.2018.06.008",
        )
    ],
    notes="Simplified Christoffel-function proxy based on Toeplitz moment matrices.",
)
def christoffel_darboux_spectral_estimation(
    y: Array,
    order: int = 30,
    grid_size: int = 1024,
    reg: float = 1e-8,
) -> SpectralEstimateResult:
    """
    Simplified Christoffel-Darboux kernel estimate on the unit circle.

    For full OPUC-based research use, one would explicitly construct orthogonal
    polynomials on the unit circle; this implementation keeps the API practical
    and stable for experimentation.
    """
    spec = moment_based_spectral_reconstruction(y, max_lag=order, grid_size=grid_size)
    moments = spec.moments
    T = np.empty((order + 1, order + 1), dtype=np.complex128)
    for i in range(order + 1):
        for j in range(order + 1):
            T[i, j] = moments[abs(i - j)]
    T += reg * np.eye(order + 1)
    theta = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    kernel_values = np.zeros(grid_size)
    density = np.zeros(grid_size)
    Tinv = np.linalg.inv(T)
    for idx, t in enumerate(theta):
        z = np.exp(1j * t)
        v = np.array([z**k for k in range(order + 1)], dtype=np.complex128)
        kappa = np.real(np.conj(v) @ Tinv @ v)
        kernel_values[idx] = kappa
        density[idx] = 1.0 / max(kappa, 1e-14)
    norm = np.trapezoid(density, theta)
    if norm > 0:
        density /= norm
    return SpectralEstimateResult(
        moments=moments,
        support_angles=theta,
        density=density,
        kernel_values=kernel_values,
        metadata={"order": order, "regularization": reg},
    )


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------


def build_sindy_library(X: Array, degree: int = 2, include_bias: bool = True) -> tuple[Array, list[str]]:
    X = _as_2d_samples(X)
    n, d = X.shape
    cols: list[Array] = []
    names: list[str] = []
    if include_bias:
        cols.append(np.ones((n, 1)))
        names.append("1")
    for j in range(d):
        cols.append(X[:, [j]])
        names.append(f"x{j}")
    if degree >= 2:
        from itertools import combinations_with_replacement

        for deg in range(2, degree + 1):
            for idx in combinations_with_replacement(range(d), deg):
                term = np.ones(n)
                label_parts: list[str] = []
                for j in idx:
                    term *= X[:, j]
                    label_parts.append(f"x{j}")
                cols.append(term[:, None])
                names.append("*".join(label_parts))
    return np.hstack(cols), names



def default_polynomial_edmd(X: Array, degree: int = 2, reg: float = 1e-8) -> LinearKoopmanResult:
    return edmd(X, feature_map=PolynomialFeatures(degree=degree), reg=reg)



def default_generator(X: Array, dt: float, degree: int = 2, reg: float = 1e-8) -> LinearKoopmanResult:
    return koopman_generator(X, dt=dt, feature_map=PolynomialFeatures(degree=degree), reg=reg)
