"""
Result container for CD-kernel reconstruction.

This module defines the structured result object returned by CD-kernel
reconstruction routines. The object stores the moments, Toeplitz matrix,
unit-circle grid, kernel diagonal, density proxy, and associated metadata,
and provides a convenience method for extracting dominant peaks.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from methods.cd_kernel.core.peaks import find_top_peaks

Array = np.ndarray


@dataclass
class CDKernelResult:
    moments: Array
    toeplitz: Array
    angles: Array
    circle_points: Array
    kernel_diag: Array
    density_proxy: Array
    regularization: float
    metadata: dict

    def top_peaks(self, k: int = 5, min_separation: int = 5):
        idx = find_top_peaks(self.density_proxy, k=k, min_separation=min_separation)
        return [
            {
                "index": int(i),
                "angle": float(self.angles[i]),
                "point": complex(self.circle_points[i]),
                "value": float(self.density_proxy[i]),
            }
            for i in idx
        ]