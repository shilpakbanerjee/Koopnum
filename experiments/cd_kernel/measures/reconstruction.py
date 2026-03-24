"""
Measure-side interface to the shared CD-kernel core.

This module keeps the measure-reconstruction API stable while delegating
the actual CD-kernel numerical work to the shared core implementation.
It exists to preserve the conceptual split:
    measures → moments → reconstruction
while avoiding duplicated code.
"""

from experiments.cd_kernel.core.result import CDKernelResult as CDMeasureResult
from experiments.cd_kernel.core.kernel import evaluate_cd_kernel_from_moments
from experiments.cd_kernel.core.peaks import find_top_peaks
from experiments.cd_kernel.core.normalization import (
    circle_integral as integrate_on_circle,
    normalize_density_proxy,
    unit_circle_grid,
)
from experiments.cd_kernel.measures.cesaro import (
    CesaroResult,
    cesaro_density_from_moments,
)
from experiments.cd_kernel.measures.quadrature import (
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