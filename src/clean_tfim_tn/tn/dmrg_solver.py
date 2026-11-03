"""DMRG ground-state solver for the clean open TFIM chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from clean_tfim_tn.models import CleanZZXTFIM


@dataclass(frozen=True)
class DMRGGroundState:
    """Result of a finite-MPS DMRG optimization.

    The ``state`` and ``model`` attributes are TeNPy objects. ``info`` and
    ``stage_energies`` contain separate diagnostics for the two-site and
    optional one-site stages.
    """

    L: int
    J: float
    hx: float
    bc: str
    chi_max: int
    energy: float
    state: Any
    model: Any
    info: dict[str, Any]
    stage_energies: dict[str, float]

    @property
    def energy_density(self) -> float:
        return self.energy / self.L


def _default_product_state(L: int, conserve: str | None = None) -> list[str]:
    """Return a product state compatible with the selected representation."""

    if conserve == "parity":
        return ["up"] * L
    return ["up" if i % 2 == 0 else "down" for i in range(L)]


def mps_to_dense_state(state: Any) -> np.ndarray:
    """Convert a finite TeNPy MPS into a normalized dense state vector."""

    if getattr(state, "bc", None) != "finite":
        raise ValueError("mps_to_dense_state requires a finite MPS")
    theta = state.get_theta(0, n=state.L)
    dense = np.asarray(theta.to_ndarray(), dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(dense)
    if norm == 0.0:
        raise ValueError("MPS represents a zero state")
    return dense / norm


def half_chain_entanglement_entropy(state: Any) -> float:
    """Return the von Neumann entropy across the central bond of an MPS.

    For even L the cut is L/2 | L/2; for odd L it is floor(L/2) | ceil(L/2).
    """

    L = int(getattr(state, "L", 0))
    if L < 2:
        raise ValueError("half-chain entropy requires an MPS length >= 2")
    bond_entropies = np.asarray(state.entanglement_entropy(), dtype=float)
    return float(bond_entropies[(L // 2) - 1])


def dmrg_ground_state(
    L: int,
    J: float = 1.0,
    hx: float = 0.5,
    chi_max: int = 32,
    bc: str = "open",
    *,
    initial_state: Sequence[str] | None = None,
    initial_mps: Any | None = None,
    max_sweeps: int = 20,
    max_E_err: float = 1e-10,
    max_S_err: float = 1e-8,
    svd_min: float = 1e-12,
    mixer: bool = True,
    one_site_sweeps: int = 4,
    one_site_max_E_err: float | None = None,
    one_site_max_S_err: float | None = None,
    one_site_mixer: bool = False,
    conserve: str | None = None,
) -> DMRGGroundState:
    """Approximate the clean TFIM ground state with hybrid finite DMRG.

    Parameters use the same Pauli convention as the exact solver. Only open
    boundary conditions are exposed because a finite MPS is used. The first
    stage uses two-site DMRG to build the Schmidt space; ``one_site_sweeps``
    then controls an optional faster one-site refinement stage.
    """

    if L < 2:
        raise ValueError("DMRG requires L >= 2")
    if chi_max < 1:
        raise ValueError("chi_max must be positive")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive")
    if one_site_sweeps < 0:
        raise ValueError("one_site_sweeps must be non-negative")
    if bc not in {"open", "obc"}:
        raise ValueError("DMRG currently supports only 'open' or 'obc'")
    if initial_state is not None and initial_mps is not None:
        raise ValueError("provide either initial_state or initial_mps, not both")
    if conserve not in {None, "parity"}:
        raise ValueError("conserve must be None or 'parity'")

    try:
        from tenpy.algorithms import dmrg
        from tenpy.networks.mps import MPS
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("TeNPy is required for DMRG; install physics-tenpy.") from exc

    model = CleanZZXTFIM.build(
        {
            "L": int(L),
            "J": float(J),
            "hx": float(hx),
            "bc_MPS": "finite",
            "bc_x": "open",
            "conserve": conserve,
        }
    )

    if initial_mps is not None:
        if getattr(initial_mps, "L", None) != L:
            raise ValueError("initial_mps must have length L")
        if getattr(initial_mps, "bc", None) != "finite":
            raise ValueError("initial_mps must have finite boundary conditions")
        psi = initial_mps.copy()
    else:
        product_state = (
            list(initial_state)
            if initial_state is not None
            else _default_product_state(L, conserve=conserve)
        )
        if len(product_state) != L:
            raise ValueError("initial_state must contain exactly L local states")
        psi = MPS.from_product_state(
            model.lat.mps_sites(),
            product_state,
            bc="finite",
            unit_cell_width=L,
        )

    trunc_params = {
        "chi_max": int(chi_max),
        "svd_min": float(svd_min),
    }
    two_site_params = {
        "active_sites": 2,
        "mixer": bool(mixer),
        "max_sweeps": int(max_sweeps),
        "max_E_err": float(max_E_err),
        "max_S_err": float(max_S_err),
        "trunc_params": trunc_params,
    }
    two_site_info = dmrg.run(psi, model, two_site_params)
    psi.canonical_form()
    two_site_energy = float(model.H_MPO.expectation_value(psi).real)

    one_site_info = None
    stage_energies = {"two_site": two_site_energy}
    if one_site_sweeps > 0:
        one_site_params = {
            "active_sites": 1,
            "mixer": bool(one_site_mixer),
            "max_sweeps": int(one_site_sweeps),
            "max_E_err": float(
                one_site_max_E_err
                if one_site_max_E_err is not None
                else min(max_E_err, 1e-11)
            ),
            "max_S_err": float(
                one_site_max_S_err
                if one_site_max_S_err is not None
                else min(max_S_err, 1e-9)
            ),
            "trunc_params": trunc_params,
        }
        one_site_info = dmrg.run(psi, model, one_site_params)
        psi.canonical_form()
        stage_energies["one_site"] = float(model.H_MPO.expectation_value(psi).real)

    energy = stage_energies.get("one_site", two_site_energy)

    return DMRGGroundState(
        L=int(L),
        J=float(J),
        hx=float(hx),
        bc=bc,
        chi_max=int(chi_max),
        energy=energy,
        state=psi,
        model=model,
        info={"two_site": two_site_info, "one_site": one_site_info},
        stage_energies=stage_energies,
    )
