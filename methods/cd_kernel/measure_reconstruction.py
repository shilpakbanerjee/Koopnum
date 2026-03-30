"""
Measure-side interface to the shared CD-kernel core.

This module keeps the measure-reconstruction API stable while delegating
the actual CD-kernel numerical work to the shared core implementation.
It exists to preserve the conceptual split:
    measures → moments → reconstruction
while avoiding duplicated code.
"""

from methods.common.results import CDKernelResult as CDMeasureResult
from methods.cd_kernel.core.kernel import evaluate_cd_kernel_from_moments
from methods.cd_kernel.core.peaks import find_top_peaks
from methods.cd_kernel.core.normalization import (
    circle_integral as integrate_on_circle,
    normalize_density_proxy,
    unit_circle_grid,
)
from methods.common.measures.cesaro import (
    CesaroResult,
    cesaro_density_from_moments,
)
from methods.common.measures.quadrature import (
    QuadratureResult,
    reconstruct_atomic_measure_from_moments,
    significant_atoms,
)

__all__ = [
    "CDMeasureResult",
    "CesaroResult",
    "QuadratureResult",
    "evaluate_cd_kernel_from_moments",
    "cesaro_density_from_moments",
    "reconstruct_atomic_measure_from_moments",
    "significant_atoms",
    "find_top_peaks",
    "integrate_on_circle",
    "normalize_density_proxy",
    "unit_circle_grid",
]