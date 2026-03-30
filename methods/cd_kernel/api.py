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

__all__ = [
    "evaluate_cd_kernel_from_moments",
    "atomic_mass_proxy_from_kernel",
    "ac_density_proxy_from_kernel",
    "koopman_matrix_from_moments",
    "companion_koopman_from_moments",
    "spectral_summary",
]