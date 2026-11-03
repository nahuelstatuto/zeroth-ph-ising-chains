"""Single-start DMRG solver for one random reentrant Ising realization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from clean_tfim_tn.classical import exact_classical_ground_state
from clean_tfim_tn.disorder import normalize_bc
from clean_tfim_tn.models import RandomIsingAntiferromagnet


@dataclass(frozen=True)
class RandomDMRGGroundState:
    """Result for one random-Ising DMRG candidate."""

    L: int
    J: np.ndarray
    Gamma: np.ndarray
    h: float
    bc: str
    chi_max: int
    initial_state_label: str
    energy: float
    state: Any
    model: Any
    info: dict[str, Any]
    stage_energies: dict[str, float]

    @property
    def energy_density(self) -> float:
        return self.energy / self.L


def random_ising_dmrg_ground_state(
    L: int,
    J: np.ndarray,
    Gamma: np.ndarray,
    h: float,
    chi_max: int = 64,
    bc: str = "open",
    *,
    initial_state: Sequence[str] | None = None,
    initial_state_name: str | None = None,
    initial_mps: Any | None = None,
    max_sweeps: int = 20,
    one_site_sweeps: int = 4,
    max_E_err: float = 1e-10,
    max_S_err: float = 1e-8,
    svd_min: float = 1e-12,
    mixer: bool = True,
) -> RandomDMRGGroundState:
    """Optimize one finite-chain random-Ising candidate with hybrid DMRG.

    For PBC, the closing bond is represented as a long-range term in a finite
    MPS/MPO. This is exact at the Hamiltonian level but is more demanding than
    the nearest-neighbour OBC representation.
    """

    normalized_bc = normalize_bc(bc)
    if chi_max < 1:
        raise ValueError("chi_max must be positive")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive")
    if one_site_sweeps < 0:
        raise ValueError("one_site_sweeps must be non-negative")
    if initial_state is not None and initial_mps is not None:
        raise ValueError("provide either initial_state or initial_mps, not both")

    try:
        from tenpy.algorithms import dmrg
        from tenpy.networks.mps import MPS
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("TeNPy is required for random-Ising DMRG.") from exc

    couplings = np.asarray(J, dtype=float)
    transverse_fields = np.asarray(Gamma, dtype=float)
    model = RandomIsingAntiferromagnet.build(
        {
            "L": L,
            "J": couplings,
            "Gamma": transverse_fields,
            "h": h,
            "bc": normalized_bc,
        }
    )

    if initial_mps is not None:
        if getattr(initial_mps, "L", None) != L:
            raise ValueError("initial_mps must have length L")
        if getattr(initial_mps, "bc", None) != "finite":
            raise ValueError("initial_mps must have finite boundary conditions")
        psi = initial_mps.copy()
        initial_state_label = initial_state_name or "previous_mps"
    else:
        if initial_state is None:
            classical = exact_classical_ground_state(
                L=L, J=couplings, h=h, bc=normalized_bc
            )
            product_state = classical.product_state_labels()
            initial_state_label = "classical_ground_state"
        else:
            product_state = list(initial_state)
            if len(product_state) != L:
                raise ValueError("initial_state must contain exactly L local states")
            initial_state_label = initial_state_name or "explicit_product_state"
        psi = MPS.from_product_state(
            model.lat.mps_sites(),
            product_state,
            bc="finite",
            unit_cell_width=L,
        )

    trunc_params = {"chi_max": int(chi_max), "svd_min": float(svd_min)}
    two_site_info = dmrg.run(
        psi,
        model,
        {
            "active_sites": 2,
            "mixer": bool(mixer),
            "max_sweeps": int(max_sweeps),
            "max_E_err": float(max_E_err),
            "max_S_err": float(max_S_err),
            "trunc_params": trunc_params,
        },
    )
    psi.canonical_form()
    stage_energies = {"two_site": float(model.H_MPO.expectation_value(psi).real)}

    one_site_info = None
    if one_site_sweeps > 0:
        one_site_info = dmrg.run(
            psi,
            model,
            {
                "active_sites": 1,
                "mixer": False,
                "max_sweeps": int(one_site_sweeps),
                "max_E_err": float(min(max_E_err, 1e-11)),
                "max_S_err": float(min(max_S_err, 1e-9)),
                "trunc_params": trunc_params,
            },
        )
        psi.canonical_form()
        stage_energies["one_site"] = float(model.H_MPO.expectation_value(psi).real)

    final_energy = stage_energies.get("one_site", stage_energies["two_site"])
    saved_J = np.array(couplings, copy=True)
    saved_Gamma = np.array(transverse_fields, copy=True)
    saved_J.setflags(write=False)
    saved_Gamma.setflags(write=False)
    return RandomDMRGGroundState(
        L=int(L),
        J=saved_J,
        Gamma=saved_Gamma,
        h=float(h),
        bc=normalized_bc,
        chi_max=int(chi_max),
        initial_state_label=initial_state_label,
        energy=final_energy,
        state=psi,
        model=model,
        info={"two_site": two_site_info, "one_site": one_site_info},
        stage_energies=stage_energies,
    )
