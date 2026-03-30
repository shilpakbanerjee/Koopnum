"""
Toeplitz matrix construction for CD-kernel reconstruction.

This module builds Hermitian Toeplitz matrices from moment sequences and
provides lightweight numerical diagnostics and regularization.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def build_hermitian_toeplitz(moments: Array, order: int | None = None) -> Array:
    moments = np.asarray(moments, dtype=np.complex128)
    if moments.ndim != 1:
        raise ValueError("moments must be 1D")

    max_order = len(moments) - 1
    if order is None:
        order = max_order
    if order > max_order:
        raise ValueError("order exceeds available moments")

    T = np.empty((order + 1, order + 1), dtype=np.complex128)
    for i in range(order + 1):
        for j in range(order + 1):
            diff = j - i
            if diff >= 0:
                T[i, j] = moments[diff]
            else:
                T[i, j] = np.conjugate(moments[-diff])
    return T


def regularize_toeplitz(T: Array, lam: float) -> Array:
    T = np.asarray(T, dtype=np.complex128)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be square")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    return T + lam * np.eye(T.shape[0], dtype=np.complex128)


def condition_number(T: Array) -> float:
    return float(np.linalg.cond(np.asarray(T, dtype=np.complex128)))


def is_hermitian(T: Array, atol: float = 1e-10) -> bool:
    T = np.asarray(T, dtype=np.complex128)
    return np.allclose(T, T.conj().T, atol=atol)