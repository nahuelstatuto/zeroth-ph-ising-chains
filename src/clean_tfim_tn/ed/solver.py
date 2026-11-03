"""Ground-state solvers for small clean TFIM systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh

from .hamiltonian import build_tfim_hamiltonian


@dataclass(frozen=True)
class ExactGroundState:
    L: int
    J: float
    hx: float
    bc: str
    energy: float
    state: np.ndarray


def _fix_global_phase(state: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(state)))
    phase = np.angle(state[idx])
    fixed = state * np.exp(-1j * phase)
    if fixed[idx].real < 0:
        fixed = -fixed
    return fixed


def exact_ground_state(
    L: int,
    J: float = 1.0,
    hx: float = 0.5,
    bc: str = "open",
) -> ExactGroundState:
    H = build_tfim_hamiltonian(L=L, J=J, hx=hx, bc=bc)
    dim = H.shape[0]

    if dim <= 4:
        evals, evecs = np.linalg.eigh(H.toarray())
        idx = int(np.argmin(evals))
        energy = float(evals[idx])
        state = evecs[:, idx]
    else:
        evals, evecs = eigsh(H, k=1, which="SA")
        energy = float(evals[0])
        state = evecs[:, 0]

    state = np.asarray(state, dtype=np.complex128)
    state = state / np.linalg.norm(state)
    state = _fix_global_phase(state)
    return ExactGroundState(L=L, J=float(J), hx=float(hx), bc=bc, energy=energy, state=state)
