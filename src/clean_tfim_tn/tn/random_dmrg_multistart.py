"""Multistart selection for random-Ising DMRG candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .random_dmrg_solver import RandomDMRGGroundState, random_ising_dmrg_ground_state


DEFAULT_INITIAL_STATES = (
    "classical_ground_state",
    "neel_up",
    "neel_down",
    "all_up",
    "random_z",
)


@dataclass(frozen=True)
class RandomDMRGMultistartResult:
    """Accepted candidate and diagnostics from a multistart search."""

    accepted: RandomDMRGGroundState
    candidates: tuple[RandomDMRGGroundState, ...]
    failures: dict[str, str]
    near_degenerate_labels: tuple[str, ...]
    degeneracy_tolerance: float
    overlap_tolerance: float

    @property
    def is_nearly_degenerate(self) -> bool:
        return len(self.near_degenerate_labels) > 1


def product_state_for_label(L: int, label: str, rng: np.random.Generator) -> list[str] | None:
    """Construct a named product state; ``None`` requests the classical solver."""

    if label == "classical_ground_state":
        return None
    if label == "neel_up":
        return ["up" if i % 2 == 0 else "down" for i in range(L)]
    if label == "neel_down":
        return ["down" if i % 2 == 0 else "up" for i in range(L)]
    if label == "all_up":
        return ["up"] * L
    if label == "all_down":
        return ["down"] * L
    if label == "random_z":
        return ["up" if value else "down" for value in rng.integers(0, 2, size=L)]
    raise ValueError(f"unknown initial-state label: {label}")


def random_ising_dmrg_multistart(
    L: int,
    J: np.ndarray,
    Gamma: np.ndarray,
    h: float,
    chi_max: int = 64,
    bc: str = "open",
    *,
    initial_states: Sequence[str] = DEFAULT_INITIAL_STATES,
    random_seed: int = 0,
    degeneracy_tolerance: float = 1e-9,
    overlap_tolerance: float = 1e-8,
    max_sweeps: int = 20,
    one_site_sweeps: int = 4,
    max_E_err: float = 1e-10,
    max_S_err: float = 1e-8,
    svd_min: float = 1e-12,
    mixer: bool = True,
) -> RandomDMRGMultistartResult:
    """Run named initial states and accept the lowest-energy candidate."""

    labels = tuple(initial_states)
    if not labels:
        raise ValueError("initial_states must not be empty")
    if len(set(labels)) != len(labels):
        raise ValueError("initial-state labels must be unique")
    if degeneracy_tolerance < 0.0 or overlap_tolerance < 0.0:
        raise ValueError("degeneracy and overlap tolerances must be non-negative")

    rng = np.random.default_rng(random_seed)
    candidates: list[RandomDMRGGroundState] = []
    failures: dict[str, str] = {}
    for label in labels:
        try:
            product_state = product_state_for_label(L, label, rng)
            candidate = random_ising_dmrg_ground_state(
                L=L,
                J=J,
                Gamma=Gamma,
                h=h,
                chi_max=chi_max,
                bc=bc,
                initial_state=product_state,
                initial_state_name=label,
                max_sweeps=max_sweeps,
                one_site_sweeps=one_site_sweeps,
                max_E_err=max_E_err,
                max_S_err=max_S_err,
                svd_min=svd_min,
                mixer=mixer,
            )
            candidates.append(candidate)
        except Exception as exc:  # candidates are independent by design
            failures[label] = f"{type(exc).__name__}: {exc}"

    if not candidates:
        details = "; ".join(f"{key}={value}" for key, value in failures.items())
        raise RuntimeError(f"all DMRG initial states failed: {details}")

    accepted = min(candidates, key=lambda candidate: candidate.energy)
    near_degenerate = [accepted.initial_state_label]
    for candidate in candidates:
        if candidate is accepted:
            continue
        if abs(candidate.energy - accepted.energy) > degeneracy_tolerance:
            continue
        fidelity = float(abs(accepted.state.overlap(candidate.state)) ** 2)
        if 1.0 - fidelity > overlap_tolerance:
            near_degenerate.append(candidate.initial_state_label)

    return RandomDMRGMultistartResult(
        accepted=accepted,
        candidates=tuple(candidates),
        failures=failures,
        near_degenerate_labels=tuple(near_degenerate),
        degeneracy_tolerance=float(degeneracy_tolerance),
        overlap_tolerance=float(overlap_tolerance),
    )
