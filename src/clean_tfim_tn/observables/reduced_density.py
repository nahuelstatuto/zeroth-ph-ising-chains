"""Reduced density matrices from dense finite-chain states."""

from __future__ import annotations

import numpy as np


def two_site_rdm(state: np.ndarray, L: int, i: int, j: int) -> np.ndarray:
    """Return the two-site reduced density matrix for sites `(i, j)`.

    The output basis is `|00>, |01>, |10>, |11>` with `i` before `j`.
    """

    if i == j:
        raise ValueError("two_site_rdm requires two distinct sites")
    if not (0 <= i < L and 0 <= j < L):
        raise ValueError("site indices out of range")

    psi = np.asarray(state, dtype=np.complex128).reshape(-1)
    if psi.size != 2**L:
        raise ValueError("state size must be 2**L")
    psi = psi / np.linalg.norm(psi)

    tensor = psi.reshape((2,) * L)
    moved = np.moveaxis(tensor, [i, j], [0, 1])
    matrix = moved.reshape(4, -1)
    rho = matrix @ matrix.conj().T
    return 0.5 * (rho + rho.conj().T)


def all_two_site_rdms(state: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    pairs: list[tuple[int, int]] = []
    rdms: list[np.ndarray] = []
    for i in range(L):
        for j in range(i + 1, L):
            pairs.append((i, j))
            rdms.append(two_site_rdm(state, L, i, j))
    return np.asarray(pairs, dtype=int), np.asarray(rdms, dtype=np.complex128)
