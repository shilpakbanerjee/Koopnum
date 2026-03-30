from __future__ import annotations

"""
Finite-dimensional Koopman approximation from a truncated moment sequence.

Mathematical overview
---------------------
Given a measure-preserving system and an observable f, define the moments

    m_k = <U^k f, f>,   k >= 0,

where U is the Koopman operator.

Fix an order n. On the Krylov subspace

    K_n = span{f, Uf, ..., U^{n-1}f},

the Gram matrix is Toeplitz:

    G_{ij} = <U^i f, U^j f> = m_{j-i},   0 <= i,j <= n-1,

with m_{-k} = conjugate(m_k).

The shifted cross-Gram matrix is

    S_{ij} = <U^{i+1} f, U^j f> = m_{j-i-1}.

The least-squares / Galerkin finite Koopman matrix K is determined by

    G K = S,

so that, in the Krylov basis,

    U(U^j f) ≈ sum_i K_{ij} U^i f.

Equivalently,

    K = G^{-1} S

when G is invertible. Numerically we solve the linear system instead of
forming the inverse explicitly.

Notes
-----
1. This construction depends on the observable f.
2. For continuous spectrum, eigenvalues of the finite matrix K are not
   direct point-spectrum estimators; they should be interpreted together
   with spectral-measure diagnostics.
3. The same moment data used for CD-kernel / Cesàro / quadrature
   reconstruction can be used here.
"""

from dataclasses import dataclass
import numpy as np

Array = np.ndarray


@dataclass
class KoopmanApproximationResult:
    """
    Output of finite-dimensional Koopman approximation.

    Attributes
    ----------
    order:
        Krylov dimension n.
    moments:
        Input moment sequence [m_0, ..., m_M].
    gram:
        Toeplitz Gram matrix G.
    shifted_gram:
        Shifted cross-Gram matrix S.
    koopman_matrix:
        Finite Koopman matrix K solving G K ≈ S.
    eigenvalues:
        Eigenvalues of K.
    eigenvectors:
        Right eigenvectors of K.
    singular_values:
        Singular values of K.
    metadata:
        Diagnostic metadata.
    """

    order: int
    moments: Array
    gram: Array
    shifted_gram: Array
    koopman_matrix: Array
    eigenvalues: Array
    eigenvectors: Array
    singular_values: Array
    metadata: dict

    def spectral_radius(self) -> float:
        if self.eigenvalues.size == 0:
            return 0.0
        return float(np.max(np.abs(self.eigenvalues)))

    def condition_number(self) -> float:
        return float(self.metadata["gram_condition_number"])

    def is_gram_well_conditioned(self, threshold: float = 1e10) -> bool:
        return self.condition_number() < threshold

    def summary(self) -> dict:
        return {
            "order": self.order,
            "gram_condition_number": float(self.metadata["gram_condition_number"]),
            "gram_rank": int(self.metadata["gram_rank"]),
            "spectral_radius": self.spectral_radius(),
            "frobenius_norm": float(self.metadata["koopman_frobenius_norm"]),
            "operator_2_norm": float(self.metadata["koopman_operator_2_norm"]),
            "residual_frobenius_norm": float(self.metadata["residual_frobenius_norm"]),
            "solve_method": self.metadata["solve_method"],
            "regularization": float(self.metadata["regularization"]),
        }


def _validate_moments(moments: Array, order: int | None = None) -> tuple[Array, int]:
    moments = np.asarray(moments, dtype=np.complex128)
    if moments.ndim != 1:
        raise ValueError("moments must be a 1D array")
    if len(moments) == 0:
        raise ValueError("moments must be nonempty")

    max_order = len(moments) - 1
    if order is None:
        order = max_order // 2 if max_order >= 2 else max_order

    if order <= 0:
        raise ValueError("order must be positive")
    if order > max_order:
        raise ValueError(f"order must satisfy 1 <= order <= {max_order}")

    return moments, int(order)


def _moment_at(moments: Array, k: int) -> complex:
    """
    Return m_k for k in Z using m_{-k} = conjugate(m_k).
    """
    if k >= 0:
        return complex(moments[k])
    return complex(np.conjugate(moments[-k]))


def toeplitz_gram_from_moments(moments: Array, order: int) -> Array:
    """
    Build the Toeplitz Gram matrix

        G_{ij} = <U^i f, U^j f> = m_{j-i}.
    """
    G = np.empty((order, order), dtype=np.complex128)
    for i in range(order):
        for j in range(order):
            G[i, j] = _moment_at(moments, j - i)
    return G


def shifted_gram_from_moments(moments: Array, order: int) -> Array:
    """
    Build the shifted cross-Gram matrix

        S_{ij} = <U^{i+1} f, U^j f> = m_{j-i-1}.
    """
    S = np.empty((order, order), dtype=np.complex128)
    for i in range(order):
        for j in range(order):
            S[i, j] = _moment_at(moments, j - i - 1)
    return S


def koopman_matrix_from_moments(
    moments: Array,
    order: int | None = None,
    regularization: float = 0.0,
    solve_method: str = "solve",
) -> KoopmanApproximationResult:
    """
    Construct a finite-dimensional Koopman approximation from moments.

    Parameters
    ----------
    moments:
        Truncated moment sequence [m_0, ..., m_M].
    order:
        Krylov dimension n. Uses moments up to at least n.
    regularization:
        Optional diagonal Tikhonov regularization added to the Gram matrix.
    solve_method:
        One of:
            - "solve"     : solve (G + reg I) K = S
            - "lstsq"     : least-squares solve
            - "pinv"      : K = pinv(G + reg I) @ S

    Returns
    -------
    KoopmanApproximationResult
    """
    moments, order = _validate_moments(moments, order=order)

    G = toeplitz_gram_from_moments(moments, order=order)
    S = shifted_gram_from_moments(moments, order=order)

    G_reg = G.copy()
    if regularization > 0.0:
        G_reg = G_reg + regularization * np.eye(order, dtype=np.complex128)

    solve_method = solve_method.lower().strip()

    if solve_method == "solve":
        K = np.linalg.solve(G_reg, S)
    elif solve_method == "lstsq":
        K, *_ = np.linalg.lstsq(G_reg, S, rcond=None)
    elif solve_method == "pinv":
        K = np.linalg.pinv(G_reg) @ S
    else:
        raise ValueError("solve_method must be one of: 'solve', 'lstsq', 'pinv'")

    eigenvalues, eigenvectors = np.linalg.eig(K)
    singular_values = np.linalg.svd(K, compute_uv=False)

    residual = G_reg @ K - S

    metadata = {
        "gram_condition_number": float(np.linalg.cond(G_reg)),
        "gram_rank": int(np.linalg.matrix_rank(G_reg)),
        "koopman_frobenius_norm": float(np.linalg.norm(K, ord="fro")),
        "koopman_operator_2_norm": float(np.linalg.norm(K, ord=2)),
        "residual_frobenius_norm": float(np.linalg.norm(residual, ord="fro")),
        "solve_method": solve_method,
        "regularization": float(regularization),
    }

    return KoopmanApproximationResult(
        order=order,
        moments=moments,
        gram=G,
        shifted_gram=S,
        koopman_matrix=K,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        singular_values=singular_values,
        metadata=metadata,
    )


def companion_koopman_from_moments(moments: Array, order: int | None = None) -> KoopmanApproximationResult:
    """
    Construct the shift / companion-style finite Koopman matrix in the
    non-orthonormal Krylov basis.

    In the basis {f, Uf, ..., U^{n-1}f}, the exact shift would map
        U^j f -> U^{j+1} f
    for j < n-1,
    while the last vector U^n f is projected back onto the span.

    This leads to a matrix with ones on the first subdiagonal and the final
    column determined by projection of U^n f onto the span.

    Compared with `koopman_matrix_from_moments`, this version exposes the
    explicit shift structure and can be useful for debugging / interpretation.
    """
    moments, order = _validate_moments(moments, order=order)

    G = toeplitz_gram_from_moments(moments, order=order)

    # rhs_j = <U^j f, U^n f> = m_{n-j}
    rhs = np.array([_moment_at(moments, order - j) for j in range(order)], dtype=np.complex128)
    coeffs = np.linalg.solve(G, rhs)

    K = np.zeros((order, order), dtype=np.complex128)

    # U(U^j f) = U^{j+1}f for j = 0, ..., order-2
    for j in range(order - 1):
        K[j + 1, j] = 1.0

    # projection of U^n f into span
    K[:, order - 1] = coeffs

    eigenvalues, eigenvectors = np.linalg.eig(K)
    singular_values = np.linalg.svd(K, compute_uv=False)

    residual = G @ K - shifted_gram_from_moments(moments, order=order)

    metadata = {
        "gram_condition_number": float(np.linalg.cond(G)),
        "gram_rank": int(np.linalg.matrix_rank(G)),
        "koopman_frobenius_norm": float(np.linalg.norm(K, ord="fro")),
        "koopman_operator_2_norm": float(np.linalg.norm(K, ord=2)),
        "residual_frobenius_norm": float(np.linalg.norm(residual, ord="fro")),
        "solve_method": "companion_projection",
        "regularization": 0.0,
    }

    return KoopmanApproximationResult(
        order=order,
        moments=moments,
        gram=G,
        shifted_gram=shifted_gram_from_moments(moments, order=order),
        koopman_matrix=K,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        singular_values=singular_values,
        metadata=metadata,
    )


def project_forward(coeffs: Array, koopman_matrix: Array, num_steps: int = 1) -> Array:
    """
    Propagate coefficient vector(s) forward using the finite Koopman matrix.

    Parameters
    ----------
    coeffs:
        Shape (n,) or (n, m), where n is the Krylov dimension.
    koopman_matrix:
        Finite Koopman matrix K.
    num_steps:
        Number of forward applications.

    Returns
    -------
    ndarray
        K^num_steps @ coeffs
    """
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    K = np.asarray(koopman_matrix, dtype=np.complex128)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("koopman_matrix must be square")

    if coeffs.shape[0] != K.shape[0]:
        raise ValueError("coeffs and koopman_matrix have incompatible leading dimension")

    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative")

    if num_steps == 0:
        return coeffs.copy()

    Kpow = np.linalg.matrix_power(K, num_steps)
    return Kpow @ coeffs


def sort_eigenvalues_by_modulus(eigenvalues: Array, descending: bool = True) -> Array:
    """
    Return indices sorting eigenvalues by modulus.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    order = np.argsort(np.abs(eigenvalues))
    if descending:
        order = order[::-1]
    return order


def spectral_summary(result: KoopmanApproximationResult, top_k: int = 10) -> list[dict]:
    """
    Summarize the leading eigenvalues by modulus.
    """
    idx = sort_eigenvalues_by_modulus(result.eigenvalues, descending=True)[:top_k]
    out = []
    for i in idx:
        lam = result.eigenvalues[i]
        out.append({
            "index": int(i),
            "eigenvalue": complex(lam),
            "modulus": float(np.abs(lam)),
            "argument": float(np.angle(lam)),
        })
    return out