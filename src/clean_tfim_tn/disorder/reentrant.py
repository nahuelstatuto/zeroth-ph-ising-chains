"""Reproducible disorder for the reentrant random Ising extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BoundaryCondition = Literal["open", "periodic"]
Ensemble = Literal["raw", "fixed_mean"]


def normalize_bc(bc: str) -> BoundaryCondition:
    aliases = {"open": "open", "obc": "open", "periodic": "periodic", "pbc": "periodic"}
    try:
        return aliases[bc]
    except KeyError as exc:
        raise ValueError("bc must be one of 'open', 'obc', 'periodic', or 'pbc'") from exc


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DisorderRealization:
    """Physical couplings obtained from one fixed set of base random values."""

    L: int
    seed: int
    bc: BoundaryCondition
    ensemble: Ensemble
    Gamma0: float
    u: np.ndarray
    v: np.ndarray
    J: np.ndarray
    Gamma: np.ndarray

    @property
    def has_negative_transverse_fields(self) -> bool:
        return bool(np.any(self.Gamma < 0.0))


@dataclass(frozen=True)
class DisorderBase:
    """Dimensionless random pattern reused throughout parameter scans."""

    L: int
    seed: int
    bc: BoundaryCondition
    u: np.ndarray
    v: np.ndarray

    def realize(self, Gamma0: float, ensemble: Ensemble = "fixed_mean") -> DisorderRealization:
        """Materialize ``J`` and ``Gamma`` without regenerating randomness."""

        if Gamma0 < 0.0:
            raise ValueError("Gamma0 must be non-negative")
        if ensemble not in {"raw", "fixed_mean"}:
            raise ValueError("ensemble must be 'raw' or 'fixed_mean'")

        J = 1.0 + self.u
        Gamma = float(Gamma0) * (1.0 + self.v)
        if ensemble == "fixed_mean":
            J = J - np.mean(J) + 1.5
            Gamma = Gamma - np.mean(Gamma) + 1.5 * float(Gamma0)

        return DisorderRealization(
            L=self.L,
            seed=self.seed,
            bc=self.bc,
            ensemble=ensemble,
            Gamma0=float(Gamma0),
            u=_readonly_copy(self.u),
            v=_readonly_copy(self.v),
            J=_readonly_copy(J),
            Gamma=_readonly_copy(Gamma),
        )


def generate_disorder_base(L: int, seed: int, bc: str = "open") -> DisorderBase:
    """Generate fixed ``u`` and ``v`` arrays using ``numpy.default_rng``."""

    if L < 2:
        raise ValueError("L must be at least 2")
    normalized_bc = normalize_bc(bc)
    n_couplings = L - 1 if normalized_bc == "open" else L
    rng = np.random.default_rng(seed)
    u = _readonly_copy(rng.uniform(0.0, 1.0, size=n_couplings))
    v = _readonly_copy(rng.uniform(0.0, 1.0, size=L))
    return DisorderBase(L=int(L), seed=int(seed), bc=normalized_bc, u=u, v=v)


def generate_disorder_realization(
    L: int,
    seed: int,
    Gamma0: float,
    bc: str = "open",
    ensemble: Ensemble = "fixed_mean",
) -> DisorderRealization:
    """Convenience wrapper generating and materializing one realization."""

    return generate_disorder_base(L=L, seed=seed, bc=bc).realize(
        Gamma0=Gamma0, ensemble=ensemble
    )
