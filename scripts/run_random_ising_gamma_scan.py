from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clean_tfim_tn.disorder import generate_disorder_base, normalize_bc
from clean_tfim_tn.io import (
    ensure_dir,
    inspect_mps_collection_hdf5,
    replace_with_retry,
    save_mps_collection_hdf5,
    write_json,
)
from clean_tfim_tn.tn import (
    DEFAULT_INITIAL_STATES,
    measure_random_ising_gamma_continuation,
    random_ising_dmrg_gamma_continuation_scan,
)


RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MPS_COLLECTION_FILE = "selected_mps.h5"
MPS_MANIFEST_FILE = "selected_mps_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an in-memory forward/backward DMRG scan in Gamma0 using one "
            "fixed disorder base."
        )
    )
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--h", type=float, default=2.1)
    parser.add_argument("--ensemble", choices=("raw", "fixed_mean"), default="fixed_mean")
    parser.add_argument(
        "--bc",
        choices=("open", "obc", "periodic", "pbc"),
        default="open",
        help="Physical boundary condition; PBC uses an explicit first-last MPO bond.",
    )
    parser.add_argument("--Gamma0", type=float, nargs="+", default=None)
    parser.add_argument("--Gamma0-min", type=float, default=0.0)
    parser.add_argument("--Gamma0-max", type=float, default=1.2)
    parser.add_argument("--Gamma0-points", type=int, default=25)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--initial-states", nargs="+", default=list(DEFAULT_INITIAL_STATES))
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--two-site-sweeps", type=int, default=20)
    parser.add_argument("--one-site-sweeps", type=int, default=4)
    parser.add_argument("--max-E-err", type=float, default=1e-10)
    parser.add_argument("--max-S-err", type=float, default=1e-8)
    parser.add_argument("--svd-min", type=float, default=1e-12)
    parser.add_argument("--branch-energy-tol", type=float, default=1e-8)
    parser.add_argument("--branch-infidelity-tol", type=float, default=1e-6)
    parser.add_argument("--selection-energy-tol", type=float, default=1e-12)
    parser.add_argument(
        "--save-mps",
        choices=("none", "selected"),
        default="none",
        help=(
            "Persist one selected MPS per Gamma0 point; at disagreement points "
            "both forward/backward states are also retained."
        ),
    )
    parser.add_argument("--no-mixer", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--directory-name",
        type=str,
        default=None,
        help=(
            "Optional filesystem directory name. This may be shorter than run-name "
            "to avoid the Windows MAX_PATH limit; run-name remains in the metadata."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/random_ising_gamma_scan")
    )
    return parser.parse_args()


def build_gamma0_grid(args: argparse.Namespace) -> np.ndarray:
    if args.Gamma0 is not None:
        values = np.asarray(args.Gamma0, dtype=float)
    else:
        if args.Gamma0_points < 1:
            raise ValueError("Gamma0-points must be positive")
        if args.Gamma0_max < args.Gamma0_min:
            raise ValueError("Gamma0-max must be greater than or equal to Gamma0-min")
        values = np.linspace(args.Gamma0_min, args.Gamma0_max, args.Gamma0_points)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Gamma0 values must be non-empty and finite")
    if np.any(values < 0.0):
        raise ValueError("Gamma0 values must be non-negative")
    return np.unique(values)


def validate_run_name(run_name: str | None) -> str | None:
    if run_name is None:
        return None
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "run-name must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return run_name


def collect_mps_for_persistence(
    result: Any, measurements: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Select persistent states using the same branch labels as the observables."""

    if len(measurements) != len(result.Gamma0_values):
        raise ValueError("measurement count does not match the Gamma0 grid")
    selected: dict[str, Any] = {}
    disagreement_forward: dict[str, Any] = {}
    disagreement_backward: dict[str, Any] = {}
    records = []
    for index, (value, measurement, forward, backward, comparison) in enumerate(zip(
        result.Gamma0_values,
        measurements,
        result.forward,
        result.backward,
        result.comparisons,
    )):
        key = f"gamma0_{index:04d}"
        if measurement.selected_branch == "forward":
            candidate = forward
        elif measurement.selected_branch == "backward":
            candidate = backward
        else:
            raise ValueError(f"unknown selected branch {measurement.selected_branch!r}")
        selected[key] = candidate.state
        if comparison.disagrees:
            disagreement_forward[key] = forward.state
            disagreement_backward[key] = backward.state
        records.append({
            "point_index": index,
            "state_key": key,
            "Gamma0": float(value),
            "selected_branch": measurement.selected_branch,
            "selected_energy": float(candidate.energy),
            "selected_bond_dimensions": [int(chi) for chi in candidate.state.chi],
            "selected_max_bond_dimension": int(max(candidate.state.chi)),
            "branch_disagreement": bool(comparison.disagrees),
            "forward_state_key_if_disagrees": key if comparison.disagrees else None,
            "backward_state_key_if_disagrees": key if comparison.disagrees else None,
        })
    return selected, disagreement_forward, disagreement_backward, records


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    # Replacing the final suffix keeps long Windows paths below MAX_PATH.  In
    # particular, ``selected_mps_manifest.json.tmp`` crossed 260 characters in
    # production while ``selected_mps_manifest.tmp`` does not.
    temporary = path.with_suffix(".tmp")
    write_json(temporary, payload)
    replace_with_retry(temporary, path)


def save_npz_with_directory_retry(path: Path, **arrays: Any) -> None:
    """Write an NPZ robustly when OneDrive transiently removes a new folder.

    A cloud-files hydration race was observed on Worker 1: ``mkdir`` returned
    successfully, but the just-created seed directory was absent when NumPy
    opened the output a moment later.  The DMRG result is already in memory at
    this point, so retrying only this write is safe and avoids rerunning a seed.
    """
    last_error: FileNotFoundError | None = None
    for attempt in range(8):
        ensure_dir(path.parent)
        try:
            np.savez_compressed(path, **arrays)
            return
        except FileNotFoundError as exc:
            last_error = exc
            if attempt == 7:
                break
            time.sleep(0.1 * (attempt + 1))
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    if args.L < 2:
        raise ValueError("L must be an integer at least 2")
    if args.chi < 1:
        raise ValueError("chi must be positive")
    if not np.isfinite(args.h):
        raise ValueError("h must be finite")
    run_name = validate_run_name(args.run_name)
    requested_directory_name = validate_run_name(args.directory_name)
    gamma0_values = build_gamma0_grid(args)
    bc = normalize_bc(args.bc)
    base = generate_disorder_base(L=args.L, seed=args.seed, bc=bc)

    if requested_directory_name is not None:
        directory_name = requested_directory_name
    else:
        directory_name = f"L{args.L}_seed{args.seed}_h_{args.h:.12g}_chi{args.chi}"
        if run_name is not None:
            directory_name = f"{directory_name}_{run_name}"
    output_dir = args.output_root / directory_name
    npz_path = output_dir / "continuation_summary.npz"
    json_path = output_dir / "continuation_summary.json"
    mps_path = output_dir / MPS_COLLECTION_FILE
    mps_manifest_path = output_dir / MPS_MANIFEST_FILE
    output_paths = (npz_path, json_path, mps_path, mps_manifest_path)
    if any(path.exists() for path in output_paths) and not args.overwrite:
        raise FileExistsError(
            f"results already exist in {output_dir}; pass --overwrite to replace them"
        )

    print(
        f"Running Gamma0 continuation: L={args.L}, seed={args.seed}, "
        f"h={args.h:g}, bc={bc}, points={len(gamma0_values)}"
    )
    result = random_ising_dmrg_gamma_continuation_scan(
        disorder_base=base,
        Gamma0_values=gamma0_values,
        h=args.h,
        chi_max=args.chi,
        ensemble=args.ensemble,
        initial_states=args.initial_states,
        random_seed=args.random_seed,
        branch_energy_tolerance=args.branch_energy_tol,
        branch_infidelity_tolerance=args.branch_infidelity_tol,
        max_sweeps=args.two_site_sweeps,
        one_site_sweeps=args.one_site_sweeps,
        max_E_err=args.max_E_err,
        max_S_err=args.max_S_err,
        svd_min=args.svd_min,
        mixer=not args.no_mixer,
    )
    measurements = measure_random_ising_gamma_continuation(
        result, selection_energy_tolerance=args.selection_energy_tol
    )

    forward_energy = np.asarray([point.energy for point in result.forward])
    backward_energy = np.asarray([point.energy for point in result.backward])
    energy_difference = np.asarray([point.absolute_energy_difference for point in result.comparisons])
    fidelity = np.asarray([point.fidelity for point in result.comparisons])
    disagreement = np.asarray([point.disagrees for point in result.comparisons])
    forward_max_chi = np.asarray([max(point.state.chi) for point in result.forward])
    backward_max_chi = np.asarray([max(point.state.chi) for point in result.backward])
    selected_branch = np.asarray([point.selected_branch for point in measurements])
    half_chain_entropy = np.asarray([point.selected.half_chain_entropy for point in measurements])
    czz = np.stack([point.selected.Czz for point in measurements])
    cstag = np.stack([point.selected.Cstag for point in measurements])
    distance = np.stack([point.selected.D for point in measurements])
    gamma_by_point = np.stack([sample.Gamma for sample in result.realizations])

    disagreement_forward_czz = np.full_like(czz, np.nan)
    disagreement_backward_czz = np.full_like(czz, np.nan)
    disagreement_forward_entropy = np.full(len(measurements), np.nan)
    disagreement_backward_entropy = np.full(len(measurements), np.nan)
    for index, point in enumerate(measurements):
        if point.forward_if_disagrees is not None:
            disagreement_forward_czz[index] = point.forward_if_disagrees.Czz
            disagreement_forward_entropy[index] = point.forward_if_disagrees.half_chain_entropy
        if point.backward_if_disagrees is not None:
            disagreement_backward_czz[index] = point.backward_if_disagrees.Czz
            disagreement_backward_entropy[index] = point.backward_if_disagrees.half_chain_entropy

    save_npz_with_directory_retry(
        npz_path,
        Gamma0_values=result.Gamma0_values,
        h=np.asarray(result.h),
        u=base.u,
        v=base.v,
        J=result.realizations[0].J,
        Gamma=gamma_by_point,
        forward_energy=forward_energy,
        backward_energy=backward_energy,
        forward_energy_density=forward_energy / args.L,
        backward_energy_density=backward_energy / args.L,
        absolute_energy_difference=energy_difference,
        branch_fidelity=fidelity,
        branch_disagreement=disagreement,
        forward_max_bond_dimension=forward_max_chi,
        backward_max_bond_dimension=backward_max_chi,
        selected_branch=selected_branch,
        half_chain_entropy=half_chain_entropy,
        Czz=czz,
        Cstag=cstag,
        D=distance,
        disagreement_forward_Czz=disagreement_forward_czz,
        disagreement_backward_Czz=disagreement_backward_czz,
        disagreement_forward_half_chain_entropy=disagreement_forward_entropy,
        disagreement_backward_half_chain_entropy=disagreement_backward_entropy,
    )

    mps_records: list[dict[str, Any]] = []
    if args.save_mps == "selected":
        selected_states, disagreement_forward_states, disagreement_backward_states, mps_records = (
            collect_mps_for_persistence(result, measurements)
        )
        mps_metadata = {
            "schema_version": 1,
            "model": "random_ising_antiferromagnet",
            "scan_parameter": "Gamma0",
            "L": args.L,
            "seed": args.seed,
            "h": args.h,
            "ensemble": args.ensemble,
            "bc": bc,
            "chi_max": args.chi,
            "run_name": run_name,
            "Gamma0_values": result.Gamma0_values,
            "selection_energy_tolerance": args.selection_energy_tol,
            "point_records": mps_records,
        }
        save_mps_collection_hdf5(
            mps_path,
            selected_states,
            mps_metadata,
            disagreement_forward_states=disagreement_forward_states,
            disagreement_backward_states=disagreement_backward_states,
        )
        collection_info = inspect_mps_collection_hdf5(mps_path)
        atomic_write_json(mps_manifest_path, {
            **mps_metadata,
            "mps_file": MPS_COLLECTION_FILE,
            "mps_file_size_bytes": mps_path.stat().st_size,
            "collection_info": collection_info,
        })
    else:
        mps_path.unlink(missing_ok=True)
        mps_manifest_path.unlink(missing_ok=True)

    rows = []
    for index, value in enumerate(result.Gamma0_values):
        rows.append({
            "Gamma0": float(value),
            "forward_energy": forward_energy[index],
            "backward_energy": backward_energy[index],
            "absolute_energy_difference": energy_difference[index],
            "branch_fidelity": fidelity[index],
            "branch_disagreement": disagreement[index],
            "forward_initial_state_label": result.forward[index].initial_state_label,
            "backward_initial_state_label": result.backward[index].initial_state_label,
            "forward_max_bond_dimension": forward_max_chi[index],
            "backward_max_bond_dimension": backward_max_chi[index],
            "selected_branch": str(selected_branch[index]),
            "selected_energy": measurements[index].selected.energy,
            "half_chain_entropy": half_chain_entropy[index],
            "selected_mps_key": (
                mps_records[index]["state_key"] if mps_records else None
            ),
        })
    write_json(json_path, {
        "metadata": {
            "model": "random_ising_antiferromagnet",
            "scan_parameter": "Gamma0",
            "L": args.L,
            "seed": args.seed,
            "h": args.h,
            "ensemble": args.ensemble,
            "bc": bc,
            "chi_max": args.chi,
            "run_name": run_name,
            "Gamma0_values": result.Gamma0_values,
            "initial_states": args.initial_states,
            "multistart_random_seed": args.random_seed,
            "two_site_sweeps": args.two_site_sweeps,
            "one_site_sweeps": args.one_site_sweeps,
            "max_E_err": args.max_E_err,
            "max_S_err": args.max_S_err,
            "svd_min": args.svd_min,
            "mixer": not args.no_mixer,
            "branch_energy_tolerance": args.branch_energy_tol,
            "branch_infidelity_tolerance": args.branch_infidelity_tol,
            "selection_energy_tolerance": args.selection_energy_tol,
            "mps_save_policy": args.save_mps,
            "mps_collection_file": (
                MPS_COLLECTION_FILE if args.save_mps == "selected" else None
            ),
            "mps_manifest_file": (
                MPS_MANIFEST_FILE if args.save_mps == "selected" else None
            ),
            "has_branch_disagreement": result.has_branch_disagreement,
            "forward_seed_failures": result.forward_seed.failures,
            "backward_seed_failures": result.backward_seed.failures,
            "forward_near_degenerate_labels": result.forward_seed.near_degenerate_labels,
            "backward_near_degenerate_labels": result.backward_seed.near_degenerate_labels,
        },
        "disorder_base": {"u": base.u, "v": base.v, "J": result.realizations[0].J},
        "Gamma_by_point": gamma_by_point,
        "rows": rows,
    })
    print(f"Summary written to: {output_dir}")


if __name__ == "__main__":
    main()
