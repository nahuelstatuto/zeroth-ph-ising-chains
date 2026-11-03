"""Forward/backward continuation scans for the random Ising model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from clean_tfim_tn.disorder import normalize_bc

from .random_dmrg_multistart import (
    DEFAULT_INITIAL_STATES,
    RandomDMRGMultistartResult,
    random_ising_dmrg_multistart,
)
from .random_dmrg_solver import RandomDMRGGroundState, random_ising_dmrg_ground_state


@dataclass(frozen=True)
class RandomDMRGBranchComparison:
    """Agreement diagnostics between both continuation branches at one field."""

    h: float
    absolute_energy_difference: float
    fidelity: float
    energy_tolerance: float
    infidelity_tolerance: float

    @property
    def disagrees(self) -> bool:
        return (
            self.absolute_energy_difference > self.energy_tolerance
            or 1.0 - self.fidelity > self.infidelity_tolerance
        )


@dataclass(frozen=True)
class RandomDMRGContinuationResult:
    """Forward/backward candidates and their pointwise comparison."""

    h_values: np.ndarray
    forward: tuple[RandomDMRGGroundState, ...]
    backward: tuple[RandomDMRGGroundState, ...]
    comparisons: tuple[RandomDMRGBranchComparison, ...]
    forward_seed: RandomDMRGMultistartResult
    backward_seed: RandomDMRGMultistartResult

    @property
    def has_branch_disagreement(self) -> bool:
        return any(comparison.disagrees for comparison in self.comparisons)


def _validate_h_values(h_values: Sequence[float]) -> np.ndarray:
    fields = np.asarray(h_values, dtype=float)
    if fields.ndim != 1 or fields.size == 0:
        raise ValueError("h_values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(fields)):
        raise ValueError("h_values must contain only finite values")
    fields = np.sort(fields)
    if np.any(np.diff(fields) <= 0.0):
        raise ValueError("h_values must not contain duplicates")
    fields.setflags(write=False)
    return fields


def _run_branch(
    fields: np.ndarray,
    *,
    direction: str,
    L: int,
    J: np.ndarray,
    Gamma: np.ndarray,
    bc: str,
    chi_max: int,
    initial_states: Sequence[str],
    random_seed: int,
    max_sweeps: int,
    one_site_sweeps: int,
    max_E_err: float,
    max_S_err: float,
    svd_min: float,
    mixer: bool,
) -> tuple[tuple[RandomDMRGGroundState, ...], RandomDMRGMultistartResult]:
    ordered_fields = fields if direction == "forward" else fields[::-1]
    seed = random_ising_dmrg_multistart(
        L=L,
        J=J,
        Gamma=Gamma,
        bc=bc,
        h=float(ordered_fields[0]),
        chi_max=chi_max,
        initial_states=initial_states,
        random_seed=random_seed,
        max_sweeps=max_sweeps,
        one_site_sweeps=one_site_sweeps,
        max_E_err=max_E_err,
        max_S_err=max_S_err,
        svd_min=svd_min,
        mixer=mixer,
    )
    candidates = [seed.accepted]
    previous_mps = seed.accepted.state
    for field in ordered_fields[1:]:
        candidate = random_ising_dmrg_ground_state(
            L=L,
            J=J,
            Gamma=Gamma,
            bc=bc,
            h=float(field),
            chi_max=chi_max,
            initial_mps=previous_mps,
            initial_state_name=f"{direction}_continuation",
            max_sweeps=max_sweeps,
            one_site_sweeps=one_site_sweeps,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            svd_min=svd_min,
            mixer=mixer,
        )
        candidates.append(candidate)
        previous_mps = candidate.state

    if direction == "backward":
        candidates.reverse()
    return tuple(candidates), seed


def random_ising_dmrg_continuation_scan(
    L: int,
    J: np.ndarray,
    Gamma: np.ndarray,
    h_values: Sequence[float],
    chi_max: int = 64,
    *,
    bc: str = "open",
    initial_states: Sequence[str] = DEFAULT_INITIAL_STATES,
    random_seed: int = 0,
    branch_energy_tolerance: float = 1e-8,
    branch_infidelity_tolerance: float = 1e-6,
    max_sweeps: int = 20,
    one_site_sweeps: int = 4,
    max_E_err: float = 1e-10,
    max_S_err: float = 1e-8,
    svd_min: float = 1e-12,
    mixer: bool = True,
) -> RandomDMRGContinuationResult:
    """Scan increasing/decreasing ``h`` for one fixed disorder sample."""

    fields = _validate_h_values(h_values)
    normalized_bc = normalize_bc(bc)
    if branch_energy_tolerance < 0.0 or branch_infidelity_tolerance < 0.0:
        raise ValueError("branch tolerances must be non-negative")

    common = dict(
        L=L,
        J=J,
        Gamma=Gamma,
        bc=normalized_bc,
        chi_max=chi_max,
        initial_states=initial_states,
        random_seed=random_seed,
        max_sweeps=max_sweeps,
        one_site_sweeps=one_site_sweeps,
        max_E_err=max_E_err,
        max_S_err=max_S_err,
        svd_min=svd_min,
        mixer=mixer,
    )
    forward, forward_seed = _run_branch(fields, direction="forward", **common)
    backward, backward_seed = _run_branch(fields, direction="backward", **common)

    comparisons = []
    for field, forward_candidate, backward_candidate in zip(fields, forward, backward):
        fidelity = float(abs(forward_candidate.state.overlap(backward_candidate.state)) ** 2)
        comparisons.append(
            RandomDMRGBranchComparison(
                h=float(field),
                absolute_energy_difference=abs(
                    forward_candidate.energy - backward_candidate.energy
                ),
                fidelity=min(max(fidelity, 0.0), 1.0),
                energy_tolerance=float(branch_energy_tolerance),
                infidelity_tolerance=float(branch_infidelity_tolerance),
            )
        )

    return RandomDMRGContinuationResult(
        h_values=fields,
        forward=forward,
        backward=backward,
        comparisons=tuple(comparisons),
        forward_seed=forward_seed,
        backward_seed=backward_seed,
    )
