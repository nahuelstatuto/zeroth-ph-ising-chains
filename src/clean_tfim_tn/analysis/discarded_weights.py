"""Discarded Schmidt-weight calculations from exact states."""

from __future__ import annotations

import numpy as np


def schmidt_values_all_cuts(state: np.ndarray, L: int) -> list[np.ndarray]:
    psi = np.asarray(state, dtype=np.complex128).reshape(-1)
    if psi.size != 2**L:
        raise ValueError("state size must be 2**L")
    psi = psi / np.linalg.norm(psi)

    values = []
    for ell in range(1, L):
        matrix = psi.reshape(2**ell, 2 ** (L - ell))
        svals = np.linalg.svd(matrix, compute_uv=False)
        values.append(np.asarray(svals, dtype=float))
    return values


def discarded_weights_for_chi(
    schmidt_values: list[np.ndarray],
    chi: int,
) -> tuple[np.ndarray, float, float]:
    if chi < 1:
        raise ValueError("chi must be positive")
    weights = np.asarray(
        [float(np.sum(svals[int(chi) :] ** 2)) for svals in schmidt_values],
        dtype=float,
    )
    delta_max = float(np.max(weights) if weights.size else 0.0)
    return weights, delta_max, float(np.sqrt(max(delta_max, 0.0)))
