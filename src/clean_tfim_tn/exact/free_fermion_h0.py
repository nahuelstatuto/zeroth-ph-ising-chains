"""Exact ground state of the open random transverse-field Ising chain at h = 0 (free fermions).

Model (paper convention, Pauli operators, OBC)

    H = sum_i J_i sigma^z_i sigma^z_{i+1} - sum_i Gamma_i sigma^x_i ,   J_i, Gamma_i > 0.

Rotating every second spin about x maps the antiferromagnet to the ferromagnet, so the
staggered correlator Cstag_ij = (-1)^|i-j| <sigma^z_i sigma^z_j> equals the ferromagnetic
<sigma^z_i sigma^z_j>. With the Jordan-Wigner Majoranas a_j, b_j (field term i Gamma_j a_j b_j,
bond term i J_j b_j a_{j+1}) the Hamiltonian is H = i a^T T b with

    T_jj = Gamma_j ,   T_{j+1, j} = -J_j .

With the singular value decomposition T = U S V^T the ground state has <a_j b_l> = i W_jl,
W = U V^T, energy E = -sum(S), and

    Cstag(i, i + r) = det(M[i:i+r, i:i+r]) ,   M = -W[1:, :-1]^T .

The fidelity between two ground states is |<psi_1|psi_2>| = |det((W_1 + W_2) / 2)|^(1/2).
All contiguous principal minors are obtained in O(L^3) by Dodgson condensation; the
module also provides direct determinants for validation.
"""

from __future__ import annotations

import numpy as np


def majorana_state(J: np.ndarray, Gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return W = U V^T, the single-particle energies S and the ground-state energy."""
    J = np.asarray(J, float)
    Gamma = np.asarray(Gamma, float)
    L = Gamma.size
    if J.size != L - 1:
        raise ValueError("open chain needs L - 1 couplings")
    T = np.diag(Gamma)
    T[np.arange(1, L), np.arange(L - 1)] = -J
    U, S, Vt = np.linalg.svd(T)
    return U @ Vt, S, float(-S.sum())


def string_matrix(W: np.ndarray) -> np.ndarray:
    """M with Cstag(i, i + r) = det(M[i:i+r, i:i+r])."""
    return -W[1:, :-1].T


def cstag_direct(W: np.ndarray) -> np.ndarray:
    """Staggered correlation matrix from direct determinants (O(L^5), for validation)."""
    M = string_matrix(W)
    L = W.shape[0]
    C = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            C[i, j] = C[j, i] = np.linalg.det(M[i:j, i:j])
    return C


def cstag_elimination(W: np.ndarray, cutoff: float = 1e-15) -> np.ndarray:
    """Staggered correlation matrix from Gaussian elimination without pivoting.

    For each start site i the leading principal minors of M[i:, i:] are the running
    products of the pivots. They are the ferromagnetic correlations <sigma_i sigma_j>,
    positive by the Griffiths inequalities, so no pivoting is needed. Once a correlation
    falls below ``cutoff`` the remaining ones of that row are set to zero (they only
    enter the topology through D = 1 - C). Dodgson condensation is not usable here
    because non-principal connected minors of M vanish.
    """
    M = string_matrix(W)
    n = M.shape[0]
    C = np.eye(n + 1)
    for i in range(n):
        A = M[i:, i:].copy()
        det = 1.0
        for k in range(A.shape[0]):
            p = A[k, k]
            det *= p
            C[i, i + k + 1] = C[i + k + 1, i] = det
            if abs(det) < cutoff:
                break
            if k + 1 < A.shape[0]:
                A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k] / p, A[k, k + 1:])
    return C


def fidelity(W1: np.ndarray, W2: np.ndarray) -> float:
    """|<psi_1|psi_2>| between two ground states of the same chain length."""
    sign, logdet = np.linalg.slogdet(0.5 * (W1 + W2))
    return float(np.exp(0.5 * logdet)) if sign != 0 else 0.0


def log_fidelity(W1: np.ndarray, W2: np.ndarray) -> float:
    """log |<psi_1|psi_2>| (stable for large L)."""
    sign, logdet = np.linalg.slogdet(0.5 * (W1 + W2))
    return 0.5 * float(logdet) if sign != 0 else -np.inf


def majorana_state_modes(J: np.ndarray, Gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """W, S and the singular vectors (u0, v0) of the lowest mode."""
    J = np.asarray(J, float)
    Gamma = np.asarray(Gamma, float)
    L = Gamma.size
    T = np.diag(Gamma)
    T[np.arange(1, L), np.arange(L - 1)] = -J
    U, S, Vt = np.linalg.svd(T)
    return U @ Vt, S, U[:, -1], Vt[-1]


def log_fidelity_parity(W1: np.ndarray, W2: np.ndarray, u2: np.ndarray, v2: np.ndarray) -> float:
    """log fidelity with the parity of the lowest mode of state 2 chosen consistently.

    In the ordered phase of a finite open chain the lowest single-particle energy is
    exponentially small and can fall below machine precision. The relative sign of its
    singular vectors, i.e. the fermion parity of the computed ground state, is then
    arbitrary, and consecutive states can land in opposite sectors (overlap zero). The
    two choices are W2 and W2 - 2 u2 v2^T. Taking the larger overlap selects the sector
    continuously connected to state 1; when the mode is gapped, the flipped choice is an
    excited state whose overlap is much smaller, so the maximum is the ground-state value.
    """
    return max(log_fidelity(W1, W2), log_fidelity(W1, W2 - 2.0 * np.outer(u2, v2)))
