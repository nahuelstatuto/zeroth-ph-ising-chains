"""Schatten trace-norm error utilities."""

from __future__ import annotations

import numpy as np


def trace_norm(matrix: np.ndarray, hermitian: bool = True) -> float:
    """Return the Schatten 1-norm, not the entrywise matrix L1 norm."""

    mat = np.asarray(matrix, dtype=np.complex128)
    if hermitian:
        vals = np.linalg.eigvalsh(0.5 * (mat + mat.conj().T))
        return float(np.sum(np.abs(vals)))
    return float(np.sum(np.linalg.svd(mat, compute_uv=False)))


def local_trace_norm_errors(
    approx_rdms: np.ndarray,
    exact_rdms: np.ndarray,
) -> tuple[np.ndarray, float]:
    if approx_rdms.shape != exact_rdms.shape:
        raise ValueError("approx_rdms and exact_rdms must have the same shape")
    errors = np.empty(approx_rdms.shape[0], dtype=float)
    for n in range(approx_rdms.shape[0]):
        errors[n] = trace_norm(approx_rdms[n] - exact_rdms[n], hermitian=True)
    return errors, float(np.max(errors) if errors.size else 0.0)
