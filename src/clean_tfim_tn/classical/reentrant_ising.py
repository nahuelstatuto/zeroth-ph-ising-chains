"""Exact classical solver for finite open or periodic random Ising chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clean_tfim_tn.disorder import normalize_bc


@dataclass(frozen=True)
class ClassicalGroundState:
    """Minimum-energy classical spin configuration."""

    spins: np.ndarray
    energy: float

    def product_state_labels(self) -> list[str]:
        """Return TeNPy labels in the physical Sigmaz basis."""

        return ["up" if spin == 1 else "down" for spin in self.spins]


def classical_energy(
    spins: np.ndarray,
    J: np.ndarray,
    h: float,
    bc: str = "open",
) -> float:
    """Evaluate ``sum J_i s_i s_{i+1} - h sum s_i`` for OBC or PBC."""

    configuration = np.asarray(spins, dtype=int)
    couplings = np.asarray(J, dtype=float)
    if configuration.ndim != 1 or not np.all(np.isin(configuration, (-1, 1))):
        raise ValueError("spins must be a one-dimensional array containing only -1 and +1")
    normalized_bc = normalize_bc(bc)
    n_bonds = configuration.size - 1 if normalized_bc == "open" else configuration.size
    if couplings.shape != (n_bonds,):
        raise ValueError(f"J must have shape ({n_bonds},)")
    if not np.all(np.isfinite(couplings)) or np.any(couplings <= 0.0):
        raise ValueError("J values must be finite and positive")
    if not np.isfinite(h) or h < 0.0:
        raise ValueError("h must be a finite non-negative number")
    interaction = np.sum(
        couplings[: configuration.size - 1] * configuration[:-1] * configuration[1:]
    )
    if normalized_bc == "periodic":
        interaction += couplings[-1] * configuration[-1] * configuration[0]
    return float(interaction - h * np.sum(configuration))


def exact_classical_ground_state(
    L: int,
    J: np.ndarray,
    h: float = 0.0,
    bc: str = "open",
) -> ClassicalGroundState:
    """Solve the classical random Ising chain by dynamic programming.

    Complexity is linear in ``L``. Degenerate ties are resolved
    deterministically by preferring spin ``-1`` in the local ``argmin``.
    """

    if L < 2:
        raise ValueError("L must be at least 2")
    normalized_bc = normalize_bc(bc)
    couplings = np.asarray(J, dtype=float)
    n_bonds = L - 1 if normalized_bc == "open" else L
    if couplings.shape != (n_bonds,):
        raise ValueError(f"J must have shape ({n_bonds},)")
    if not np.all(np.isfinite(couplings)) or np.any(couplings <= 0.0):
        raise ValueError("J values must be finite and positive")
    if not np.isfinite(h) or h < 0.0:
        raise ValueError("h must be a finite non-negative number")

    spin_values = np.array([-1, 1], dtype=int)
    first_indices = range(2) if normalized_bc == "periodic" else (None,)
    best_energy = np.inf
    best_spins: np.ndarray | None = None

    for fixed_first_index in first_indices:
        if fixed_first_index is None:
            previous_energies = -float(h) * spin_values.astype(float)
        else:
            previous_energies = np.full(2, np.inf, dtype=float)
            previous_energies[fixed_first_index] = (
                -float(h) * spin_values[fixed_first_index]
            )
        backpointers = np.full((L, 2), -1, dtype=np.int8)

        for site in range(1, L):
            current_energies = np.empty(2, dtype=float)
            for current_index, current_spin in enumerate(spin_values):
                candidates = (
                    previous_energies
                    + couplings[site - 1] * spin_values * current_spin
                    - float(h) * current_spin
                )
                best_previous = int(np.argmin(candidates))
                current_energies[current_index] = candidates[best_previous]
                backpointers[site, current_index] = best_previous
            previous_energies = current_energies

        final_energies = previous_energies.copy()
        if fixed_first_index is not None:
            final_energies += (
                couplings[-1]
                * spin_values
                * spin_values[fixed_first_index]
            )
        final_index = int(np.argmin(final_energies))
        energy = float(final_energies[final_index])

        spins = np.empty(L, dtype=int)
        spins[-1] = spin_values[final_index]
        for site in range(L - 1, 0, -1):
            final_index = int(backpointers[site, final_index])
            spins[site - 1] = spin_values[final_index]

        if energy < best_energy:
            best_energy = energy
            best_spins = spins

    assert best_spins is not None
    best_spins.setflags(write=False)
    return ClassicalGroundState(spins=best_spins, energy=float(best_energy))
