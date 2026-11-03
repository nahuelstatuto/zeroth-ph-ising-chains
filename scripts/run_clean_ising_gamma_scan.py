from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clean_tfim_tn.io.clean_ising_gamma_scan import (  # noqa: E402
    MPS_COLLECTION_FILE, MPS_MANIFEST_FILE, SUMMARY_JSON, SUMMARY_NPZ,
    atomic_write_json, atomic_write_npz,
)
from clean_tfim_tn.io.clean_ising_mps_hdf5 import (  # noqa: E402
    inspect_clean_ising_mps_collection, save_clean_ising_mps_collection,
)
from clean_tfim_tn.disorder import normalize_bc  # noqa: E402
from clean_tfim_tn.tn.clean_ising_gamma_continuation import (  # noqa: E402
    DEFAULT_CLEAN_INITIAL_STATES, clean_ising_gamma_continuation_scan,
    select_and_measure_clean_ising_gamma,
)


NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a clean mixed-field Ising continuation in uniform Gamma."
    )
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--bc", choices=("open", "obc", "periodic", "pbc"), default="open")
    parser.add_argument("--Gamma", type=float, nargs="+", default=None)
    parser.add_argument("--Gamma-min", type=float, default=0.0)
    parser.add_argument("--Gamma-max", type=float, default=1.5)
    parser.add_argument("--Gamma-points", type=int, default=31)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--initial-states", nargs="+", default=list(DEFAULT_CLEAN_INITIAL_STATES))
    parser.add_argument("--two-site-sweeps", type=int, default=20)
    parser.add_argument("--one-site-sweeps", type=int, default=4)
    parser.add_argument("--max-E-err", type=float, default=1e-10)
    parser.add_argument("--max-S-err", type=float, default=1e-8)
    parser.add_argument("--svd-min", type=float, default=1e-12)
    parser.add_argument("--branch-energy-tol", type=float, default=1e-8)
    parser.add_argument("--branch-infidelity-tol", type=float, default=1e-6)
    parser.add_argument("--selection-energy-tol", type=float, default=1e-12)
    parser.add_argument("--no-mixer", action="store_true")
    parser.add_argument("--save-mps", choices=("none", "selected"), default="selected")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--directory-name",
        default=None,
        help="Safe output-directory basename; metadata still records --run-name.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("results/clean_ising_gamma_scan"))
    return parser.parse_args()


def gamma_grid(args: argparse.Namespace) -> np.ndarray:
    if args.Gamma is None:
        if args.Gamma_points < 1 or args.Gamma_max < args.Gamma_min:
            raise ValueError("invalid Gamma interval or point count")
        values = np.linspace(args.Gamma_min, args.Gamma_max, args.Gamma_points)
    else:
        values = np.asarray(args.Gamma, dtype=float)
    if values.size < 1 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Gamma values must be non-empty, finite and non-negative")
    return np.unique(values)


def validate_args(args: argparse.Namespace) -> None:
    if args.L < 2:
        raise ValueError("L must be an integer at least 2")
    if not np.isfinite(args.J) or args.J <= 0.0:
        raise ValueError("J must be finite and positive")
    if not np.isfinite(args.h) or args.h < 0.0:
        raise ValueError("h must be finite and non-negative")
    if args.chi < 1 or args.two_site_sweeps < 1 or args.one_site_sweeps < 0:
        raise ValueError("invalid chi or sweep counts")
    if args.run_name is not None and not NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("run-name contains unsupported characters")
    if args.directory_name is not None and not NAME_PATTERN.fullmatch(args.directory_name):
        raise ValueError("directory-name contains unsupported characters")
    if not args.initial_states or len(set(args.initial_states)) != len(args.initial_states):
        raise ValueError("initial-states must be non-empty and unique")
    allowed = set(DEFAULT_CLEAN_INITIAL_STATES)
    unknown = set(args.initial_states).difference(allowed)
    if unknown:
        raise ValueError(f"unknown deterministic initial states: {sorted(unknown)}")
    tolerances = np.asarray([
        args.max_E_err, args.max_S_err, args.svd_min,
        args.branch_energy_tol, args.branch_infidelity_tol,
        args.selection_energy_tol,
    ])
    if np.any(~np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("DMRG and branch tolerances must be finite and non-negative")


def collect_mps(result: Any, measured: Any):
    selected, forward_extra, backward_extra, records = {}, {}, {}, []
    for index, (value, point, forward, backward, comparison) in enumerate(zip(
        result.Gamma_values, measured, result.forward, result.backward, result.comparisons
    )):
        key = f"gamma_{index:04d}"
        candidate = forward if point.selected_branch == "forward" else backward
        selected[key] = candidate.state
        if comparison.disagrees:
            forward_extra[key], backward_extra[key] = forward.state, backward.state
        records.append({
            "point_index": index,
            "state_key": key,
            "Gamma": float(value),
            "selected_branch": point.selected_branch,
            "selected_energy": float(candidate.energy),
            "selected_bond_dimensions": [int(x) for x in candidate.state.chi],
            "selected_max_bond_dimension": int(max(candidate.state.chi)),
            "branch_disagreement": bool(comparison.disagrees),
            "forward_state_key_if_disagrees": key if comparison.disagrees else None,
            "backward_state_key_if_disagrees": key if comparison.disagrees else None,
        })
    return selected, forward_extra, backward_extra, records


def main() -> None:
    args = parse_args()
    validate_args(args)
    bc = normalize_bc(args.bc)
    gamma = gamma_grid(args)
    default_name = f"L{args.L}_h_{args.h:.12g}_J{args.J:.12g}_chi{args.chi}"
    if args.run_name:
        default_name += f"_{args.run_name}"
    name = args.directory_name or default_name
    output = args.output_root / name
    targets = [output / item for item in (
        SUMMARY_NPZ, SUMMARY_JSON, MPS_COLLECTION_FILE, MPS_MANIFEST_FILE
    )]
    if any(path.exists() for path in targets) and not args.overwrite:
        raise FileExistsError(f"results already exist in {output}; pass --overwrite")

    print(
        f"Running clean Gamma continuation: L={args.L}, h={args.h:g}, "
        f"points={gamma.size}, chi={args.chi}, bc={bc}"
    )
    result = clean_ising_gamma_continuation_scan(
        L=args.L, J=args.J, h=args.h, Gamma_values=gamma, bc=bc, chi_max=args.chi,
        initial_states=args.initial_states,
        branch_energy_tolerance=args.branch_energy_tol,
        branch_infidelity_tolerance=args.branch_infidelity_tol,
        max_sweeps=args.two_site_sweeps, one_site_sweeps=args.one_site_sweeps,
        max_E_err=args.max_E_err, max_S_err=args.max_S_err,
        svd_min=args.svd_min, mixer=not args.no_mixer,
    )
    measured = select_and_measure_clean_ising_gamma(
        result, selection_energy_tolerance=args.selection_energy_tol
    )
    forward_energy = np.asarray([point.energy for point in result.forward])
    backward_energy = np.asarray([point.energy for point in result.backward])
    energy_difference = np.asarray([point.absolute_energy_difference for point in result.comparisons])
    fidelity = np.asarray([point.fidelity for point in result.comparisons])
    disagreement = np.asarray([point.disagrees for point in result.comparisons])
    forward_max_chi = np.asarray([max(point.state.chi) for point in result.forward])
    backward_max_chi = np.asarray([max(point.state.chi) for point in result.backward])
    selected_branch = np.asarray([point.selected_branch for point in measured])
    selected_energy = np.asarray([point.selected.energy for point in measured])
    selected_max_chi = np.asarray([max(point.selected_state.state.chi) for point in measured])
    half_entropy = np.asarray([point.selected.half_chain_entropy for point in measured])
    entropies = np.stack([
        np.asarray(point.selected_state.state.entanglement_entropy(), dtype=float)
        for point in measured
    ])
    bond_dimensions = np.stack([
        np.asarray(point.selected_state.state.chi, dtype=int) for point in measured
    ])
    czz = np.stack([point.selected.Czz for point in measured])
    cstag = np.stack([point.selected.Cstag for point in measured])
    distance = np.stack([point.selected.D for point in measured])
    disagreement_forward_czz = np.full_like(czz, np.nan)
    disagreement_backward_czz = np.full_like(czz, np.nan)
    disagreement_forward_entropy = np.full(gamma.size, np.nan)
    disagreement_backward_entropy = np.full(gamma.size, np.nan)
    for index, point in enumerate(measured):
        if point.forward_if_disagrees is not None:
            disagreement_forward_czz[index] = point.forward_if_disagrees.Czz
            disagreement_forward_entropy[index] = point.forward_if_disagrees.half_chain_entropy
        if point.backward_if_disagrees is not None:
            disagreement_backward_czz[index] = point.backward_if_disagrees.Czz
            disagreement_backward_entropy[index] = point.backward_if_disagrees.half_chain_entropy

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(
        output / SUMMARY_NPZ,
        Gamma_values=gamma, h=np.asarray(args.h), J=np.asarray(args.J),
        J_values=np.full(args.L - 1 if bc == "open" else args.L, args.J),
        Gamma_by_point=np.broadcast_to(gamma[:, None], (gamma.size, args.L)).copy(),
        forward_energy=forward_energy, backward_energy=backward_energy,
        forward_energy_density=forward_energy / args.L,
        backward_energy_density=backward_energy / args.L,
        absolute_energy_difference=energy_difference, branch_fidelity=fidelity,
        branch_disagreement=disagreement,
        forward_max_bond_dimension=forward_max_chi,
        backward_max_bond_dimension=backward_max_chi,
        selected_branch=selected_branch, selected_energy=selected_energy,
        selected_energy_density=selected_energy / args.L,
        selected_max_bond_dimension=selected_max_chi,
        half_chain_entropy=half_entropy, entanglement_entropy=entropies,
        bond_dimensions=bond_dimensions, Czz=czz, Cstag=cstag, D=distance,
        disagreement_forward_Czz=disagreement_forward_czz,
        disagreement_backward_Czz=disagreement_backward_czz,
        disagreement_forward_half_chain_entropy=disagreement_forward_entropy,
        disagreement_backward_half_chain_entropy=disagreement_backward_entropy,
    )

    records = []
    if args.save_mps == "selected":
        selected, forward_extra, backward_extra, records = collect_mps(result, measured)
        mps_metadata = {
            "schema_version": 1, "model": "clean_mixed_field_ising",
            "disorder": "none", "scan_parameter": "Gamma", "L": args.L,
            "J": args.J, "h": args.h, "bc": bc, "chi_max": args.chi,
            "conserve": None, "run_name": args.run_name,
            "Gamma_values": gamma, "selection_energy_tolerance": args.selection_energy_tol,
            "point_records": records,
        }
        save_clean_ising_mps_collection(
            output / MPS_COLLECTION_FILE, selected, mps_metadata,
            disagreement_forward_states=forward_extra,
            disagreement_backward_states=backward_extra,
        )
        info = inspect_clean_ising_mps_collection(output / MPS_COLLECTION_FILE)
        atomic_write_json(output / MPS_MANIFEST_FILE, {
            **mps_metadata, "mps_file": MPS_COLLECTION_FILE,
            "mps_file_size_bytes": (output / MPS_COLLECTION_FILE).stat().st_size,
            "collection_info": info,
        })
    else:
        (output / MPS_COLLECTION_FILE).unlink(missing_ok=True)
        (output / MPS_MANIFEST_FILE).unlink(missing_ok=True)

    rows = []
    for index, value in enumerate(gamma):
        rows.append({
            "Gamma": float(value), "forward_energy": forward_energy[index],
            "backward_energy": backward_energy[index],
            "absolute_energy_difference": energy_difference[index],
            "branch_fidelity": fidelity[index], "branch_disagreement": disagreement[index],
            "forward_initial_state_label": result.forward[index].initial_state_label,
            "backward_initial_state_label": result.backward[index].initial_state_label,
            "forward_max_bond_dimension": forward_max_chi[index],
            "backward_max_bond_dimension": backward_max_chi[index],
            "selected_branch": str(selected_branch[index]),
            "selected_energy": selected_energy[index],
            "half_chain_entropy": half_entropy[index],
            "selected_mps_key": records[index]["state_key"] if records else None,
        })
    metadata = {
        "model": "clean_mixed_field_ising", "disorder": "none",
        "scan_parameter": "Gamma", "L": args.L, "J": args.J, "h": args.h,
        "bc": bc, "chi_max": args.chi, "conserve": None,
        "run_name": args.run_name, "Gamma_values": gamma,
        "initial_states": args.initial_states,
        "two_site_sweeps": args.two_site_sweeps,
        "one_site_sweeps": args.one_site_sweeps,
        "max_E_err": args.max_E_err, "max_S_err": args.max_S_err,
        "svd_min": args.svd_min, "mixer": not args.no_mixer,
        "branch_energy_tolerance": args.branch_energy_tol,
        "branch_infidelity_tolerance": args.branch_infidelity_tol,
        "selection_energy_tolerance": args.selection_energy_tol,
        "mps_save_policy": args.save_mps,
        "mps_collection_file": MPS_COLLECTION_FILE if records else None,
        "mps_manifest_file": MPS_MANIFEST_FILE if records else None,
        "has_branch_disagreement": result.has_branch_disagreement,
        "forward_endpoint_failures": result.forward_endpoint.failures,
        "backward_endpoint_failures": result.backward_endpoint.failures,
        "forward_near_degenerate_labels": result.forward_endpoint.near_degenerate_labels,
        "backward_near_degenerate_labels": result.backward_endpoint.near_degenerate_labels,
    }
    atomic_write_json(output / SUMMARY_JSON, {"metadata": metadata, "rows": rows})
    print(f"Summary written to: {output}")


if __name__ == "__main__":
    main()
