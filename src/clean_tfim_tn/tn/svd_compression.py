"""Exact dense-state compression by sequential SVD truncation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SVDCompressionResult:
    chi: int
    tensors: tuple[np.ndarray, ...]
    state: np.ndarray
    raw_norm: float
    kept_bond_dimensions: tuple[int, ...]


def _reconstruct_state(tensors: tuple[np.ndarray, ...]) -> np.ndarray:
    state = tensors[0][0, :, :]
    for tensor in tensors[1:]:
        state = np.tensordot(state, tensor, axes=([-1], [0]))
    state = np.squeeze(state, axis=-1)
    return state.reshape(-1)


def compress_dense_state_svd(state: np.ndarray, L: int, chi: int) -> SVDCompressionResult:
    """Compress an exact state with a left-to-right TT-SVD truncation."""

    if chi < 1:
        raise ValueError("chi must be positive")

    psi = np.asarray(state, dtype=np.complex128).reshape(-1)
    if psi.size != 2**L:
        raise ValueError("state size must be 2**L")
    psi = psi / np.linalg.norm(psi)

    tensors: list[np.ndarray] = []
    kept_dims: list[int] = []
    left_rank = 1
    remainder = psi.reshape(1, -1)

    for _site in range(L - 1):
        matrix = remainder.reshape(left_rank * 2, -1)
        u, svals, vh = np.linalg.svd(matrix, full_matrices=False)
        keep = min(int(chi), svals.size)
        u = u[:, :keep]
        svals = svals[:keep]
        vh = vh[:keep, :]
        tensors.append(u.reshape(left_rank, 2, keep))
        kept_dims.append(keep)
        remainder = svals[:, None] * vh
        left_rank = keep

    tensors.append(remainder.reshape(left_rank, 2, 1))
    raw_state = _reconstruct_state(tuple(tensors))
    raw_norm = float(np.linalg.norm(raw_state))
    if raw_norm == 0.0:
        raise ValueError("SVD truncation produced a zero state")
    normalized_state = raw_state / raw_norm
    return SVDCompressionResult(
        chi=int(chi),
        tensors=tuple(tensors),
        state=normalized_state,
        raw_norm=raw_norm,
        kept_bond_dimensions=tuple(kept_dims),
    )


def dense_state_to_tenpy_mps(state: np.ndarray, L: int):  # pragma: no cover - optional backend
    """Construct a finite TeNPy MPS from a dense state when TeNPy is available."""

    try:
        from tenpy.networks.mps import MPS
        from tenpy.networks.site import SpinHalfSite
    except ImportError as exc:
        raise ImportError("Install physics-tenpy to construct a TeNPy MPS.") from exc

    sites = [SpinHalfSite(conserve=None) for _ in range(L)]
    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * L)
    return MPS.from_full(sites, tensor, bc="finite")
