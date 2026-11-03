"""Observable extraction from random-Ising continuation candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clean_tfim_tn.observables import distance_from_staggered, staggered_zz_matrix

from .dmrg_solver import half_chain_entanglement_entropy
from .random_dmrg_continuation import RandomDMRGContinuationResult
from .random_dmrg_solver import RandomDMRGGroundState


@dataclass(frozen=True)
class RandomIsingMPSObservables:
    """Observables measured on one random-Ising MPS candidate."""

    energy: float
    half_chain_entropy: float
    Czz: np.ndarray
    Cstag: np.ndarray
    D: np.ndarray


@dataclass(frozen=True)
class RandomIsingContinuationObservables:
    """Selected observables plus branch-resolved data at disagreement points."""

    h: float
    selected_branch: str
    selected: RandomIsingMPSObservables
    forward_if_disagrees: RandomIsingMPSObservables | None
    backward_if_disagrees: RandomIsingMPSObservables | None


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


def random_ising_mps_observables(
    candidate: RandomDMRGGroundState,
) -> RandomIsingMPSObservables:
    """Measure entropy and raw/staggered ZZ matrices on one candidate."""

    czz = np.asarray(
        candidate.state.correlation_function(
            "Sigmaz", "Sigmaz", hermitian=True
        ).real,
        dtype=float,
    )
    czz = 0.5 * (czz + czz.T)
    np.fill_diagonal(czz, 1.0)
    cstag = staggered_zz_matrix(czz)
    distance = distance_from_staggered(cstag)
    return RandomIsingMPSObservables(
        energy=float(candidate.energy),
        half_chain_entropy=half_chain_entanglement_entropy(candidate.state),
        Czz=_readonly(czz),
        Cstag=_readonly(cstag),
        D=_readonly(distance),
    )


def measure_random_ising_continuation(
    result: RandomDMRGContinuationResult,
    *,
    selection_energy_tolerance: float = 1e-12,
) -> tuple[RandomIsingContinuationObservables, ...]:
    """Select the lower-energy branch and measure each continuation point."""

    if selection_energy_tolerance < 0.0:
        raise ValueError("selection_energy_tolerance must be non-negative")
    measurements = []
    for field, forward, backward, comparison in zip(
        result.h_values, result.forward, result.backward, result.comparisons
    ):
        if backward.energy < forward.energy - selection_energy_tolerance:
            selected_branch = "backward"
            selected_candidate = backward
        else:
            selected_branch = "forward"
            selected_candidate = forward

        selected = random_ising_mps_observables(selected_candidate)
        forward_observables = None
        backward_observables = None
        if comparison.disagrees:
            forward_observables = (
                selected
                if selected_candidate is forward
                else random_ising_mps_observables(forward)
            )
            backward_observables = (
                selected
                if selected_candidate is backward
                else random_ising_mps_observables(backward)
            )
        measurements.append(
            RandomIsingContinuationObservables(
                h=float(field),
                selected_branch=selected_branch,
                selected=selected,
                forward_if_disagrees=forward_observables,
                backward_if_disagrees=backward_observables,
            )
        )
    return tuple(measurements)
