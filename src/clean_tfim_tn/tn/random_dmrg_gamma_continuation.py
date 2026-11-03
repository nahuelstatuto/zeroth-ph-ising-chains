"""Forward/backward continuation in Gamma0 for the random Ising model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from clean_tfim_tn.disorder.reentrant import DisorderBase, DisorderRealization, Ensemble

from .random_dmrg_multistart import (
    DEFAULT_INITIAL_STATES,
    RandomDMRGMultistartResult,
    random_ising_dmrg_multistart,
)
from .random_dmrg_solver import RandomDMRGGroundState, random_ising_dmrg_ground_state


@dataclass(frozen=True)
class RandomDMRGGammaBranchComparison:
    """Agreement diagnostics between both branches at one ``Gamma0``."""

    Gamma0: float
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
class RandomDMRGGammaContinuationResult:
    """Forward/backward candidates for a scan over ``Gamma0``."""

    Gamma0_values: np.ndarray
    realizations: tuple[DisorderRealization, ...]
    h: float
    forward: tuple[RandomDMRGGroundState, ...]
    backward: tuple[RandomDMRGGroundState, ...]
    comparisons: tuple[RandomDMRGGammaBranchComparison, ...]
    forward_seed: RandomDMRGMultistartResult
    backward_seed: RandomDMRGMultistartResult

    @property
    def has_branch_disagreement(self) -> bool:
        return any(comparison.disagrees for comparison in self.comparisons)


def _validate_gamma0_values(values: Sequence[float]) -> np.ndarray:
    gamma0_values = np.asarray(values, dtype=float)
    if gamma0_values.ndim != 1 or gamma0_values.size == 0:
        raise ValueError("Gamma0_values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(gamma0_values)):
        raise ValueError("Gamma0_values must contain only finite values")
    if np.any(gamma0_values < 0.0):
        raise ValueError("Gamma0_values must be non-negative")
    gamma0_values = np.sort(gamma0_values)
    if np.any(np.diff(gamma0_values) <= 0.0):
        raise ValueError("Gamma0_values must not contain duplicates")
    gamma0_values.setflags(write=False)
    return gamma0_values


def _run_gamma_branch(
    realizations: Sequence[DisorderRealization],
    *,
    direction: str,
    h: float,
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
    ordered = realizations if direction == "forward" else realizations[::-1]
    first = ordered[0]
    seed = random_ising_dmrg_multistart(
        L=first.L, J=first.J, Gamma=first.Gamma, h=h, chi_max=chi_max,
        bc=first.bc,
        initial_states=initial_states, random_seed=random_seed,
        max_sweeps=max_sweeps, one_site_sweeps=one_site_sweeps,
        max_E_err=max_E_err, max_S_err=max_S_err, svd_min=svd_min, mixer=mixer,
    )
    candidates = [seed.accepted]
    previous_mps = seed.accepted.state
    for realization in ordered[1:]:
        candidate = random_ising_dmrg_ground_state(
            L=realization.L, J=realization.J, Gamma=realization.Gamma, h=h,
            chi_max=chi_max, bc=realization.bc, initial_mps=previous_mps,
            initial_state_name=f"{direction}_gamma_continuation",
            max_sweeps=max_sweeps, one_site_sweeps=one_site_sweeps,
            max_E_err=max_E_err, max_S_err=max_S_err, svd_min=svd_min, mixer=mixer,
        )
        candidates.append(candidate)
        previous_mps = candidate.state
    if direction == "backward":
        candidates.reverse()
    return tuple(candidates), seed


def random_ising_dmrg_gamma_continuation_scan(
    disorder_base: DisorderBase,
    Gamma0_values: Sequence[float],
    h: float,
    chi_max: int = 64,
    *,
    ensemble: Ensemble = "fixed_mean",
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
) -> RandomDMRGGammaContinuationResult:
    """Scan ``Gamma0`` in both directions using one fixed disorder base."""

    values = _validate_gamma0_values(Gamma0_values)
    if not np.isfinite(h):
        raise ValueError("h must be finite")
    if branch_energy_tolerance < 0.0 or branch_infidelity_tolerance < 0.0:
        raise ValueError("branch tolerances must be non-negative")
    realizations = tuple(
        disorder_base.realize(Gamma0=float(value), ensemble=ensemble) for value in values
    )
    common = dict(
        h=float(h), chi_max=chi_max, initial_states=initial_states,
        random_seed=random_seed, max_sweeps=max_sweeps,
        one_site_sweeps=one_site_sweeps, max_E_err=max_E_err,
        max_S_err=max_S_err, svd_min=svd_min, mixer=mixer,
    )
    forward, forward_seed = _run_gamma_branch(realizations, direction="forward", **common)
    backward, backward_seed = _run_gamma_branch(realizations, direction="backward", **common)
    comparisons = []
    for value, forward_candidate, backward_candidate in zip(values, forward, backward):
        fidelity = float(abs(forward_candidate.state.overlap(backward_candidate.state)) ** 2)
        comparisons.append(RandomDMRGGammaBranchComparison(
            Gamma0=float(value),
            absolute_energy_difference=abs(forward_candidate.energy - backward_candidate.energy),
            fidelity=min(max(fidelity, 0.0), 1.0),
            energy_tolerance=float(branch_energy_tolerance),
            infidelity_tolerance=float(branch_infidelity_tolerance),
        ))
    return RandomDMRGGammaContinuationResult(
        Gamma0_values=values, realizations=realizations, h=float(h),
        forward=forward, backward=backward, comparisons=tuple(comparisons),
        forward_seed=forward_seed, backward_seed=backward_seed,
    )
