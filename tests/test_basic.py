from __future__ import annotations

import numpy as np

from koopman_toolkit import dmd, default_polynomial_edmd, get_algorithm_registry, moment_based_spectral_reconstruction


def test_dmd_shapes():
    X = np.column_stack([np.cos(np.linspace(0, 10, 200)), np.sin(np.linspace(0, 10, 200))])
    res = dmd(X, rank=2)
    assert res.operator.shape == (2, 2)
    assert len(res.eigenvalues) == 2


def test_registry_populated():
    reg = get_algorithm_registry()
    assert "dmd" in reg
    assert "edmd" in reg


def test_moment_estimate_nonnegative_density():
    y = np.cos(np.linspace(0, 10, 400))
    res = moment_based_spectral_reconstruction(y, max_lag=20, grid_size=256)
    assert np.all(res.density >= -1e-12)
