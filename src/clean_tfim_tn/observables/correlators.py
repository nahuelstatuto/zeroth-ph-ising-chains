"""Raw and staggered Pauli-ZZ correlators."""

from __future__ import annotations

import numpy as np


def _z_eigenvalue_table(L: int) -> np.ndarray:
    states = np.arange(2**L, dtype=np.uint64)
    bits = np.empty((2**L, L), dtype=np.int8)
    for i in range(L):
        shift = L - 1 - i
        bits[:, i] = (states >> shift) & 1
    return 1.0 - 2.0 * bits


def zz_correlator_matrix(state: np.ndarray, L: int) -> np.ndarray:
    """Return raw `Czz[i, j] = <sigma^z_i sigma^z_j>`.

    This is not connected; no one-point functions are subtracted.
    """

    psi = np.asarray(state, dtype=np.complex128).reshape(-1)
    if psi.size != 2**L:
        raise ValueError("state size must be 2**L")
    norm = np.linalg.norm(psi)
    if norm == 0.0:
        raise ValueError("state must be nonzero")
    probs = np.abs(psi / norm) ** 2
    zvals = _z_eigenvalue_table(L)
    czz = (zvals * probs[:, None]).T @ zvals
    np.fill_diagonal(czz, 1.0)
    return np.asarray(czz.real, dtype=float)


def staggered_zz_matrix(czz: np.ndarray) -> np.ndarray:
    L = czz.shape[0]
    if czz.shape != (L, L):
        raise ValueError("czz must be a square matrix")
    idx = np.arange(L)
    phase = (-1.0) ** np.abs(idx[:, None] - idx[None, :])
    return phase * czz


def distance_from_staggered(cstag: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(cstag, dtype=float)
