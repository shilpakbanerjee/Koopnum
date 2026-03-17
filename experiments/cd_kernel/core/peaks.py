"""
Peak detection utilities for CD-kernel density proxies.

This module contains simple peak-finding routines used to identify dominant
local maxima in reconstructed density proxies, particularly in pure-point or
mixed spectral settings.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def find_top_peaks(density: Array, k: int = 5, min_separation: int = 5) -> Array:
    density = np.asarray(density, dtype=float)
    n = density.size
    candidates = []

    if n < 3:
        return np.array([], dtype=int)

    for i in range(1, n - 1):
        if density[i] >= density[i - 1] and density[i] >= density[i + 1]:
            candidates.append(i)

    if not candidates:
        return np.array([], dtype=int)

    candidates = sorted(candidates, key=lambda i: density[i], reverse=True)

    selected = []
    for idx in candidates:
        if all(abs(idx - j) > min_separation for j in selected):
            selected.append(idx)
        if len(selected) >= k:
            break

    return np.array(sorted(selected), dtype=int)