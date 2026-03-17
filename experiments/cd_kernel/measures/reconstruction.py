"""
Measure-side interface to the shared CD-kernel core.

This module keeps the measure-reconstruction API stable while delegating
the actual CD-kernel numerical work to the shared core implementation.
It exists to preserve the conceptual split:
    measures → moments → reconstruction
while avoiding duplicated code.
"""

from __future__ import annotations

from experiments.cd_kernel.core.result import CDKernelResult as CDMeasureResult
from experiments.cd_kernel.core.kernel import evaluate_cd_kernel_from_moments
from experiments.cd_kernel.core.peaks import find_top_peaks
from experiments.cd_kernel.core.normalization import (
    circle_integral as integrate_on_circle,
    normalize_density_proxy,
    unit_circle_grid,
)

__all__ = [
    "CDMeasureResult",
    "evaluate_cd_kernel_from_moments",
    "find_top_peaks",
    "integrate_on_circle",
    "normalize_density_proxy",
    "unit_circle_grid",
]