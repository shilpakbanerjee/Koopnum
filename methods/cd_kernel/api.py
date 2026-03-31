from __future__ import annotations

"""
Public API for CD-kernel-based spectral-measure and Koopman routines.

This module provides a thin stable front-door for runners so they do not
need to import internal implementation modules directly.
"""

from methods.cd_kernel.core.kernel import (
    evaluate_cd_kernel_from_moments,
    atomic_mass_proxy_from_kernel,
    ac_density_proxy_from_kernel,
)
from methods.cd_kernel.koopman.koopman import (
    koopman_matrix_from_moments,
    companion_koopman_from_moments,
    spectral_summary,
)

from methods.common.measures.cesaro import cesaro_density_from_moments
from methods.common.measures.quadrature import reconstruct_atomic_measure_from_moments

def run_all_measure_methods_from_moments(
    moments,
    *,
    order=None,
    grid_size: int = 2048,
    cd_normalize_density: bool = True,
    cesaro_clip_negative: bool = True,
    cesaro_normalize_mass: bool = True,
    quadrature_nodes=None,
    quadrature_mass_constraint_weight: float = 1.0,
    quadrature_normalize_mass: bool = True,
):
    """
    Run the currently supported measure-reconstruction methods from a common
    moment sequence.

    Returns
    -------
    dict
        {
            "cd": cd_result,
            "cesaro": cesaro_result,
            "quadrature": quad_result,
        }
    """
    cd_result = evaluate_cd_kernel_from_moments(
        moments,
        order=order,
        grid_size=grid_size,
        normalize_density=cd_normalize_density,
    )

    cesaro_result = cesaro_density_from_moments(
        moments,
        order=order,
        grid_size=grid_size,
        clip_negative=cesaro_clip_negative,
        normalize_mass=cesaro_normalize_mass,
    )

    quad_result = reconstruct_atomic_measure_from_moments(
        moments,
        order=order,
        num_nodes=quadrature_nodes,
        mass_constraint_weight=quadrature_mass_constraint_weight,
        normalize_mass=quadrature_normalize_mass,
    )

    return {
        "cd": cd_result,
        "cesaro": cesaro_result,
        "quadrature": quad_result,
    }

__all__ = [
    "evaluate_cd_kernel_from_moments",
    "atomic_mass_proxy_from_kernel",
    "ac_density_proxy_from_kernel",
    "koopman_matrix_from_moments",
    "companion_koopman_from_moments",
    "spectral_summary",
    "run_all_measure_methods_from_moments",
]

