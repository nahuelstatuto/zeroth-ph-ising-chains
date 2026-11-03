"""Sparse exact-diagonalization Hamiltonians for clean TFIM chains."""

from __future__ import annotations

import numpy as np
from scipy import sparse

SIGMA_X = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
SIGMA_Z = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
IDENTITY_2 = sparse.identity(2, format="csr", dtype=float)


def _many_body_operator(local_ops: list[sparse.spmatrix]) -> sparse.csr_matrix:
    op = local_ops[0]
    for local in local_ops[1:]:
        op = sparse.kron(op, local, format="csr")
    return op.tocsr()


def build_tfim_hamiltonian(
    L: int,
    J: float = 1.0,
    hx: float = 0.5,
    bc: str = "open",
) -> sparse.csr_matrix:
    """Build `H = J sum ZZ - hx sum X` using Pauli matrices.

    Site `0` is the leftmost tensor factor in the computational basis.
    """

    if L < 1:
        raise ValueError("L must be positive")
    if bc not in {"open", "obc", "periodic", "pbc"}:
        raise ValueError("bc must be one of 'open', 'obc', 'periodic', or 'pbc'")

    dim = 2**L
    H = sparse.csr_matrix((dim, dim), dtype=float)

    n_bonds = L - 1 if bc in {"open", "obc"} else L
    for i in range(n_bonds):
        j = (i + 1) % L
        ops = [IDENTITY_2] * L
        ops[i] = SIGMA_Z
        ops[j] = SIGMA_Z
        H = H + float(J) * _many_body_operator(ops)

    for i in range(L):
        ops = [IDENTITY_2] * L
        ops[i] = SIGMA_X
        H = H - float(hx) * _many_body_operator(ops)

    return H.tocsr()
