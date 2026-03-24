"""
Quadrature-style atomic weak reconstruction from moments.

This module implements a practical version of the Section 4.2.2
moment-matching quadrature approximation from Korda–Putinar–Mezić.

We fix atom locations on a uniform angular grid and solve for
nonnegative weights gamma_j such that

    m_k ~= sum_j gamma_j exp(i k eta_j),   k = 0, ..., N

in least-squares sense, subject to gamma_j >= 0.

This yields a purely atomic approximation of the spectral measure.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from scipy.optimize import lsq_linear
except Exception:  # pragma: no cover
    lsq_linear = None

Array = np.ndarray


@dataclass
class QuadratureResult:
    moments: Array
    order: int
    node_angles: Array
    weights: Array
    moment_fit: Array
    cdf_grid: Array
    cdf_values: Array
    metadata: dict


def _validate_moments(moments: Array, order: int | None = None) -> tuple[Array, int]:
    moments = np.asarray(moments, dtype=np.complex128)
    if moments.ndim != 1:
        raise ValueError("moments must be a 1D array")
    if len(moments) == 0:
        raise ValueError("moments must be nonempty")

    max_order = len(moments) - 1
    if order is None:
        order = max_order
    if order < 0 or order > max_order:
        raise ValueError(f"order must satisfy 0 <= order <= {max_order}")

    return moments, int(order)


def uniform_quadrature_nodes(
    num_nodes: int,
) -> Array:
    """
    Uniform nodes eta_j in [0, 2pi), j = 0, ..., num_nodes-1.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    return np.linspace(0.0, 2.0 * np.pi, num_nodes, endpoint=False)


def moment_matrix_from_nodes(
    node_angles: Array,
    order: int,
) -> Array:
    """
    Build A with entries
        A[k, j] = exp(i k eta_j),   k = 0, ..., order.
    """
    node_angles = np.asarray(node_angles, dtype=float)
    ks = np.arange(order + 1, dtype=int)[:, None]
    return np.exp(1j * ks * node_angles[None, :])


def reconstruct_atomic_measure_from_moments(
    moments: Array,
    order: int | None = None,
    num_nodes: int | None = None,
    node_angles: Array | None = None,
    mass_constraint_weight: float = 1.0,
    normalize_mass: bool = True,
) -> QuadratureResult:
    """
    Fit a nonnegative atomic measure on a fixed grid of nodes.

    Parameters
    ----------
    moments:
        Array [m_0, ..., m_M].
    order:
        Use moments up to order N.
    num_nodes:
        Number of uniform nodes if node_angles is not supplied.
        A natural default is 10 * (order + 1), echoing the paper's practical setup.
    node_angles:
        Optional explicit node locations in radians.
    mass_constraint_weight:
        Extra weight placed on the k=0 equation.
    normalize_mass:
        Renormalize the recovered weights to total mass Re(m_0).

    Returns
    -------
    QuadratureResult
    """
    moments, order = _validate_moments(moments, order=order)

    if node_angles is None:
        if num_nodes is None:
            num_nodes = 10 * (order + 1)
        node_angles = uniform_quadrature_nodes(num_nodes)
    else:
        node_angles = np.asarray(node_angles, dtype=float)
        num_nodes = len(node_angles)

    A_complex = moment_matrix_from_nodes(node_angles, order=order)
    b_complex = moments[: order + 1]

    # Convert complex system A gamma ~= b into real stacked system:
    #
    # [ Re A ] gamma ~= [ Re b ]
    # [ Im A ]          [ Im b ]
    #
    A_real = np.vstack([np.real(A_complex), np.imag(A_complex)])
    b_real = np.concatenate([np.real(b_complex), np.imag(b_complex)])

    # Add an explicit mass row rather than just rescaling the first equation.
    if mass_constraint_weight > 0:
        mass_row = np.ones((1, num_nodes), dtype=float)
        mass_rhs = np.array([float(np.real(moments[0]))], dtype=float)

        A_real = np.vstack([A_real, mass_constraint_weight * mass_row])
        b_real = np.concatenate([b_real, mass_constraint_weight * mass_rhs])
    gamma = _solve_nonnegative_least_squares(A_real, b_real)

    gamma = np.maximum(gamma, 0.0)

    total_mass_target = float(np.real(moments[0]))
    if normalize_mass and total_mass_target > 0:
        recovered_mass = float(np.sum(gamma))
        if recovered_mass > 0:
            gamma = gamma * (total_mass_target / recovered_mass)

    moment_fit = A_complex @ gamma
    residual = b_complex - moment_fit

    cdf_grid, cdf_values = atomic_cdf(node_angles, gamma)

    return QuadratureResult(
        moments=moments,
        order=order,
        node_angles=node_angles,
        weights=gamma,
        moment_fit=moment_fit,
        cdf_grid=cdf_grid,
        cdf_values=cdf_values,
        metadata={
            "method": "quadrature_uniform_grid",
            "num_nodes": int(num_nodes),
            "mass_target": total_mass_target,
            "mass_recovered": float(np.sum(gamma)),
            "l2_residual_real_system": float(np.linalg.norm(A_real @ gamma - b_real)),
            "max_abs_moment_residual": float(np.max(np.abs(residual))),
            "mean_abs_moment_residual": float(np.mean(np.abs(residual))),
            "mass_constraint_weight": float(mass_constraint_weight),
            "normalize_mass": normalize_mass,
        },
    )


def atomic_cdf(
    node_angles: Array,
    weights: Array,
) -> tuple[Array, Array]:
    """
    Return the atomic CDF on sorted node locations.
    """
    node_angles = np.asarray(node_angles, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if node_angles.shape != weights.shape:
        raise ValueError("node_angles and weights must have the same shape")

    order = np.argsort(node_angles)
    grid = node_angles[order]
    w = weights[order]
    cdf = np.cumsum(w)

    return grid, cdf


def significant_atoms(
    result: QuadratureResult,
    tol: float = 1e-10,
) -> list[dict]:
    """
    Extract atoms with weights above tolerance.
    """
    keep = result.weights > tol
    angles = result.node_angles[keep]
    weights = result.weights[keep]

    order = np.argsort(angles)
    angles = angles[order]
    weights = weights[order]

    return [
        {
            "angle": float(theta),
            "weight": float(w),
            "point": complex(np.exp(1j * theta)),
        }
        for theta, w in zip(angles, weights)
    ]


def _solve_nonnegative_least_squares(A: Array, b: Array) -> Array:
    """
    Solve min ||A x - b||_2 subject to x >= 0.

    Strategy:
    1. Column-scale the system to reduce conditioning issues.
    2. Try scipy.optimize.lsq_linear with method='bvls' (often more robust than TRF
       for dense bounded least squares of moderate size).
    3. Fall back to scipy.optimize.nnls if needed.

    Returns
    -------
    x : ndarray
        Nonnegative solution vector.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if b.ndim != 1:
        raise ValueError("b must be 1D")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b have incompatible shapes")

    # Column scaling: solve A D^{-1} y ~= b, x = D^{-1} y
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 1e-14, col_norms, 1.0)
    A_scaled = A / col_norms[None, :]

    # Try bounded-variable least squares first.
    if lsq_linear is not None:
        try:
            sol = lsq_linear(
                A_scaled,
                b,
                bounds=(0.0, np.inf),
                method="bvls",
                tol=1e-10,
                max_iter=2000,
                verbose=0,
            )
            if sol.success and sol.x is not None:
                x_scaled = np.asarray(sol.x, dtype=float)
                return x_scaled / col_norms
        except Exception:
            pass

        # Secondary attempt with TRF, but looser tolerances / more iterations.
        try:
            sol = lsq_linear(
                A_scaled,
                b,
                bounds=(0.0, np.inf),
                method="trf",
                tol=1e-10,
                lsmr_tol="auto",
                max_iter=4000,
                verbose=0,
            )
            if sol.success and sol.x is not None:
                x_scaled = np.asarray(sol.x, dtype=float)
                return x_scaled / col_norms
        except Exception:
            pass

    # Fallback: Lawson–Hanson NNLS
    try:
        from scipy.optimize import nnls
    except Exception as exc:
        raise RuntimeError(
            "Could not solve nonnegative least squares: "
            "lsq_linear failed and scipy.optimize.nnls is unavailable."
        ) from exc

    x_scaled, _ = nnls(A_scaled, b)
    return np.asarray(x_scaled, dtype=float) / col_norms