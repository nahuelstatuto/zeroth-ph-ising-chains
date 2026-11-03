"""Sparse Pauli Hamiltonian for the reentrant random Ising model."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from clean_tfim_tn.disorder import normalize_bc

from .hamiltonian import IDENTITY_2, SIGMA_X, SIGMA_Z, _many_body_operator


def _validated_parameter_array(
    values: np.ndarray,
    *,
    name: str,
    expected_length: int,
    strictly_positive: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.shape != (expected_length,):
        raise ValueError(f"{name} must have shape ({expected_length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if strictly_positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive")
    if not strictly_positive and np.any(array < 0.0):
        raise ValueError(f"{name} values must be non-negative")
    return array


def build_random_ising_hamiltonian(
    L: int,
    J: np.ndarray,
    Gamma: np.ndarray,
    h: float = 0.0,
    bc: str = "open",
) -> sparse.csr_matrix:
    """Build the random Ising Hamiltonian using Pauli matrices directly.

    The implemented convention is

    ``H = sum_i J_i ZZ - sum_i Gamma_i X - h sum_i Z``.
    """

    if L < 2:
        raise ValueError("L must be at least 2")
    normalized_bc = normalize_bc(bc)
    n_bonds = L - 1 if normalized_bc == "open" else L
    couplings = _validated_parameter_array(
        J, name="J", expected_length=n_bonds, strictly_positive=True
    )
    transverse_fields = _validated_parameter_array(
        Gamma, name="Gamma", expected_length=L, strictly_positive=False
    )
    if not np.isfinite(h) or h < 0.0:
        raise ValueError("h must be a finite non-negative number")

    dimension = 2**L
    hamiltonian = sparse.csr_matrix((dimension, dimension), dtype=float)

    for i, coupling in enumerate(couplings):
        j = (i + 1) % L
        operators = [IDENTITY_2] * L
        operators[i] = SIGMA_Z
        operators[j] = SIGMA_Z
        hamiltonian = hamiltonian + coupling * _many_body_operator(operators)

    for i, transverse_field in enumerate(transverse_fields):
        operators_x = [IDENTITY_2] * L
        operators_x[i] = SIGMA_X
        hamiltonian = hamiltonian - transverse_field * _many_body_operator(operators_x)

        if h != 0.0:
            operators_z = [IDENTITY_2] * L
            operators_z[i] = SIGMA_Z
            hamiltonian = hamiltonian - float(h) * _many_body_operator(operators_z)

    return hamiltonian.tocsr()
