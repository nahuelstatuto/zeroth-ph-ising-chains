"""Uniform (disorder-free) mixed-field Ising continuation in Gamma."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from clean_tfim_tn.disorder import normalize_bc

from .random_dmrg_multistart import RandomDMRGMultistartResult, random_ising_dmrg_multistart
from .random_dmrg_observables import RandomIsingMPSObservables, random_ising_mps_observables
from .random_dmrg_solver import RandomDMRGGroundState, random_ising_dmrg_ground_state


DEFAULT_CLEAN_INITIAL_STATES = (
    "classical_ground_state",
    "neel_up",
    "neel_down",
    "all_up",
    "all_down",
)


@dataclass(frozen=True)
class CleanIsingGammaBranchComparison:
    Gamma: float
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
class CleanIsingGammaContinuationResult:
    L: int
    J: float
    h: float
    bc: str
    Gamma_values: np.ndarray
    forward: tuple[RandomDMRGGroundState, ...]
    backward: tuple[RandomDMRGGroundState, ...]
    comparisons: tuple[CleanIsingGammaBranchComparison, ...]
    forward_endpoint: RandomDMRGMultistartResult
    backward_endpoint: RandomDMRGMultistartResult

    @property
    def has_branch_disagreement(self) -> bool:
        return any(comparison.disagrees for comparison in self.comparisons)


@dataclass(frozen=True)
class CleanIsingGammaSelectedPoint:
    Gamma: float
    selected_branch: str
    selected_state: RandomDMRGGroundState
    selected: RandomIsingMPSObservables
    forward_if_disagrees: RandomIsingMPSObservables | None
    backward_if_disagrees: RandomIsingMPSObservables | None


def _validated_gamma(values: Sequence[float]) -> np.ndarray:
    gamma = np.asarray(values, dtype=float)
    if gamma.ndim != 1 or gamma.size < 1 or np.any(~np.isfinite(gamma)):
        raise ValueError("Gamma values must be a non-empty finite one-dimensional grid")
    if np.any(gamma < 0.0):
        raise ValueError("Gamma values must be non-negative")
    gamma = np.sort(gamma)
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("Gamma values must be distinct")
    gamma.setflags(write=False)
    return gamma


def _uniform_arrays(
    L: int, J: float, Gamma: float, bc: str
) -> tuple[np.ndarray, np.ndarray]:
    n_bonds = L - 1 if bc == "open" else L
    return np.full(n_bonds, J, dtype=float), np.full(L, Gamma, dtype=float)


def _run_branch(
    gamma_values: np.ndarray,
    *,
    direction: str,
    L: int,
    J: float,
    h: float,
    bc: str,
    chi_max: int,
    initial_states: Sequence[str],
    solver_kwargs: dict,
) -> tuple[tuple[RandomDMRGGroundState, ...], RandomDMRGMultistartResult]:
    ordered = gamma_values if direction == "forward" else gamma_values[::-1]
    first_J, first_Gamma = _uniform_arrays(L, J, float(ordered[0]), bc)
    seed = random_ising_dmrg_multistart(
        L=L,
        J=first_J,
        Gamma=first_Gamma,
        h=h,
        bc=bc,
        chi_max=chi_max,
        initial_states=initial_states,
        random_seed=0,
        **solver_kwargs,
    )
    candidates = [seed.accepted]
    previous_mps = seed.accepted.state
    for value in ordered[1:]:
        coupling, transverse = _uniform_arrays(L, J, float(value), bc)
        candidate = random_ising_dmrg_ground_state(
            L=L,
            J=coupling,
            Gamma=transverse,
            h=h,
            bc=bc,
            chi_max=chi_max,
            initial_mps=previous_mps,
            initial_state_name=f"{direction}_clean_gamma_continuation",
            **solver_kwargs,
        )
        candidates.append(candidate)
        previous_mps = candidate.state
    if direction == "backward":
        candidates.reverse()
    return tuple(candidates), seed


def clean_ising_gamma_continuation_scan(
    *,
    L: int,
    J: float,
    h: float,
    Gamma_values: Sequence[float],
    bc: str = "open",
    chi_max: int = 64,
    initial_states: Sequence[str] = DEFAULT_CLEAN_INITIAL_STATES,
    branch_energy_tolerance: float = 1e-8,
    branch_infidelity_tolerance: float = 1e-6,
    max_sweeps: int = 20,
    one_site_sweeps: int = 4,
    max_E_err: float = 1e-10,
    max_S_err: float = 1e-8,
    svd_min: float = 1e-12,
    mixer: bool = True,
) -> CleanIsingGammaContinuationResult:
    """Scan the clean model with uniform ``J_i=J`` and ``Gamma_i=Gamma``."""

    gamma = _validated_gamma(Gamma_values)
    normalized_bc = normalize_bc(bc)
    if L < 2:
        raise ValueError("L must be an integer at least 2")
    if not np.isfinite(J) or J <= 0.0:
        raise ValueError("J must be finite and positive")
    if not np.isfinite(h) or h < 0.0:
        raise ValueError("h must be finite and non-negative")
    if chi_max < 1:
        raise ValueError("chi_max must be positive")
    if branch_energy_tolerance < 0.0 or branch_infidelity_tolerance < 0.0:
        raise ValueError("branch tolerances must be non-negative")
    solver_kwargs = {
        "max_sweeps": int(max_sweeps),
        "one_site_sweeps": int(one_site_sweeps),
        "max_E_err": float(max_E_err),
        "max_S_err": float(max_S_err),
        "svd_min": float(svd_min),
        "mixer": bool(mixer),
    }
    common = {
        "L": int(L),
        "J": float(J),
        "h": float(h),
        "bc": normalized_bc,
        "chi_max": int(chi_max),
        "initial_states": tuple(initial_states),
        "solver_kwargs": solver_kwargs,
    }
    forward, forward_seed = _run_branch(gamma, direction="forward", **common)
    backward, backward_seed = _run_branch(gamma, direction="backward", **common)
    comparisons = []
    for value, left, right in zip(gamma, forward, backward):
        fidelity = float(np.clip(abs(left.state.overlap(right.state)) ** 2, 0.0, 1.0))
        comparisons.append(CleanIsingGammaBranchComparison(
            Gamma=float(value),
            absolute_energy_difference=float(abs(left.energy - right.energy)),
            fidelity=fidelity,
            energy_tolerance=float(branch_energy_tolerance),
            infidelity_tolerance=float(branch_infidelity_tolerance),
        ))
    return CleanIsingGammaContinuationResult(
        L=int(L),
        J=float(J),
        h=float(h),
        bc=normalized_bc,
        Gamma_values=gamma,
        forward=forward,
        backward=backward,
        comparisons=tuple(comparisons),
        forward_endpoint=forward_seed,
        backward_endpoint=backward_seed,
    )


def select_and_measure_clean_ising_gamma(
    result: CleanIsingGammaContinuationResult,
    *,
    selection_energy_tolerance: float = 1e-12,
) -> tuple[CleanIsingGammaSelectedPoint, ...]:
    if selection_energy_tolerance < 0.0:
        raise ValueError("selection_energy_tolerance must be non-negative")
    measured = []
    for value, forward, backward, comparison in zip(
        result.Gamma_values, result.forward, result.backward, result.comparisons
    ):
        if backward.energy < forward.energy - selection_energy_tolerance:
            selected_branch, selected_state = "backward", backward
        else:
            selected_branch, selected_state = "forward", forward
        selected = random_ising_mps_observables(selected_state)
        forward_observables = backward_observables = None
        if comparison.disagrees:
            forward_observables = (
                selected if selected_state is forward else random_ising_mps_observables(forward)
            )
            backward_observables = (
                selected if selected_state is backward else random_ising_mps_observables(backward)
            )
        measured.append(CleanIsingGammaSelectedPoint(
            Gamma=float(value),
            selected_branch=selected_branch,
            selected_state=selected_state,
            selected=selected,
            forward_if_disagrees=forward_observables,
            backward_if_disagrees=backward_observables,
        ))
    return tuple(measured)
