"""Observable extraction for random-Ising continuation in ``Gamma0``."""

from __future__ import annotations

from dataclasses import dataclass

from .random_dmrg_gamma_continuation import RandomDMRGGammaContinuationResult
from .random_dmrg_observables import RandomIsingMPSObservables, random_ising_mps_observables


@dataclass(frozen=True)
class RandomIsingGammaContinuationObservables:
    Gamma0: float
    selected_branch: str
    selected: RandomIsingMPSObservables
    forward_if_disagrees: RandomIsingMPSObservables | None
    backward_if_disagrees: RandomIsingMPSObservables | None


def measure_random_ising_gamma_continuation(
    result: RandomDMRGGammaContinuationResult,
    *,
    selection_energy_tolerance: float = 1e-12,
) -> tuple[RandomIsingGammaContinuationObservables, ...]:
    """Select the lower-energy branch and measure every ``Gamma0`` point."""

    if selection_energy_tolerance < 0.0:
        raise ValueError("selection_energy_tolerance must be non-negative")
    measurements = []
    for value, forward, backward, comparison in zip(
        result.Gamma0_values, result.forward, result.backward, result.comparisons
    ):
        if backward.energy < forward.energy - selection_energy_tolerance:
            selected_branch, selected_candidate = "backward", backward
        else:
            selected_branch, selected_candidate = "forward", forward
        selected = random_ising_mps_observables(selected_candidate)
        forward_observables = backward_observables = None
        if comparison.disagrees:
            forward_observables = (
                selected if selected_candidate is forward
                else random_ising_mps_observables(forward)
            )
            backward_observables = (
                selected if selected_candidate is backward
                else random_ising_mps_observables(backward)
            )
        measurements.append(RandomIsingGammaContinuationObservables(
            Gamma0=float(value), selected_branch=selected_branch, selected=selected,
            forward_if_disagrees=forward_observables,
            backward_if_disagrees=backward_observables,
        ))
    return tuple(measurements)
