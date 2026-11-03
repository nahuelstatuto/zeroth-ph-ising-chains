from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clean_tfim_tn.io import inspect_mps_collection_hdf5, replace_with_retry, write_json
from clean_tfim_tn.disorder import normalize_bc
from clean_tfim_tn.tn import DEFAULT_INITIAL_STATES


RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SUMMARY_NPZ = "continuation_summary.npz"
SUMMARY_JSON = "continuation_summary.json"
MPS_COLLECTION_FILE = "selected_mps.h5"
MPS_MANIFEST_FILE = "selected_mps_manifest.json"
REQUIRED_ARRAYS = {
    "Gamma0_values",
    "h",
    "u",
    "v",
    "J",
    "Gamma",
    "forward_energy",
    "backward_energy",
    "absolute_energy_difference",
    "branch_fidelity",
    "branch_disagreement",
    "forward_max_bond_dimension",
    "backward_max_bond_dimension",
    "selected_branch",
    "half_chain_entropy",
    "Czz",
    "Cstag",
    "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable sequential campaign of fixed-h Gamma0 scans over "
            "several disorder seeds."
        )
    )
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--seed-stop", type=int, default=4,
        help="Inclusive final seed when --seeds is not provided.",
    )
    parser.add_argument(
        "--h", type=float, nargs="+", default=None,
        help=(
            "Explicit fixed-h cuts; required unless --pairs-file is used. "
            "The ordinary mode forms the Cartesian product with the seeds."
        ),
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=None,
        help=(
            "JSON specification containing a 'pairs' list of exact "
            "{'seed': int, 'h': float} jobs. This prevents an accidental "
            "Cartesian product in directed QC campaigns."
        ),
    )
    parser.add_argument("--ensemble", choices=("raw", "fixed_mean"), default="fixed_mean")
    parser.add_argument(
        "--bc",
        choices=("open", "obc", "periodic", "pbc"),
        default="open",
        help="Physical boundary condition; aliases are stored canonically.",
    )
    parser.add_argument("--Gamma0", type=float, nargs="+", default=None)
    parser.add_argument("--Gamma0-min", type=float, default=0.0)
    parser.add_argument("--Gamma0-max", type=float, default=0.5)
    parser.add_argument("--Gamma0-points", type=int, default=11)
    parser.add_argument("--chi", type=int, default=32)
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
            "Persist one selected MPS per point in each scan; disagreement "
            "points additionally retain both continuation branches."
        ),
    )
    parser.add_argument("--no-mixer", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/random_ising_gamma_ensemble"),
    )
    parser.add_argument(
        "--reuse-compatible-from",
        type=Path,
        nargs="+",
        default=[],
        help=(
            "Search these roots for complete compatible scans whose saved "
            "Gamma0 grid contains the requested grid."
        ),
    )
    parser.add_argument(
        "--rerun-incomplete",
        action="store_true",
        help="Recompute a corrupt or partial target scan using --overwrite.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_campaign_name(name: str) -> str:
    if not RUN_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "campaign-name must start with an alphanumeric character and contain "
            "only letters, numbers, '.', '_' or '-'"
        )
    return name


def build_seed_grid(args: argparse.Namespace) -> np.ndarray:
    if args.seeds is not None:
        seeds = np.asarray(args.seeds, dtype=int)
    else:
        if args.seed_stop < args.seed_start:
            raise ValueError("seed-stop must be greater than or equal to seed-start")
        seeds = np.arange(args.seed_start, args.seed_stop + 1, dtype=int)
    seeds = np.unique(seeds)
    if seeds.size == 0 or np.any(seeds < 0):
        raise ValueError("seeds must be non-empty non-negative integers")
    return seeds


def build_h_grid(args: argparse.Namespace) -> np.ndarray:
    if args.h is None:
        raise ValueError("--h is required unless --pairs-file is used")
    values = np.asarray(args.h, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("h values must be non-empty and finite")
    return np.unique(values)


def load_pair_grid(path: Path) -> list[tuple[int, float]]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("pairs-file must contain a non-empty 'pairs' list")
    pairs: list[tuple[int, float]] = []
    seen: set[tuple[int, float]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"seed", "h"}:
            raise ValueError(
                f"pairs[{index}] must contain exactly the keys 'seed' and 'h'"
            )
        seed = row["seed"]
        h = row["h"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"pairs[{index}].seed must be a non-negative integer")
        if not isinstance(h, (int, float)) or isinstance(h, bool) or not math.isfinite(h):
            raise ValueError(f"pairs[{index}].h must be finite")
        pair = (int(seed), float(h))
        canonical = (pair[0], float(format(pair[1], ".12g")))
        if canonical in seen:
            raise ValueError(f"duplicate or colliding pair in pairs-file: {pair}")
        seen.add(canonical)
        pairs.append(pair)
    return sorted(pairs, key=lambda item: (item[1], item[0]))


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


def scan_directory_name(
    L: int, seed: int, h: float, chi: int, run_name: str | None = None
) -> str:
    """Return the compact worker directory name.

    ``run_name`` remains an accepted argument for source compatibility, but the
    campaign directory already identifies the run. Repeating it here can exceed
    Windows MAX_PATH once result filenames are appended.
    """

    return f"L{L}_seed{seed}_h_{h:.12g}_chi{chi}"


def legacy_scan_directory_name(
    L: int, seed: int, h: float, chi: int, run_name: str
) -> str:
    """Return the directory convention used before compact scan names."""

    return f"{scan_directory_name(L, seed, h, chi)}_{run_name}"


def scan_directory_has_artifacts(directory: Path) -> bool:
    return any(
        (directory / name).exists()
        for name in (
            SUMMARY_NPZ,
            SUMMARY_JSON,
            MPS_COLLECTION_FILE,
            MPS_MANIFEST_FILE,
        )
    )


def _float_matches(left: Any, right: float) -> bool:
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and bool(np.isclose(value, right, rtol=5e-15, atol=0.0))


def grid_contains(saved: Any, requested: np.ndarray, tolerance: float = 1e-10) -> bool:
    try:
        saved_grid = np.asarray(saved, dtype=float)
    except (TypeError, ValueError):
        return False
    if saved_grid.ndim != 1 or saved_grid.size == 0 or not np.all(np.isfinite(saved_grid)):
        return False
    return all(np.min(np.abs(saved_grid - value)) <= tolerance for value in requested)


def expected_metadata(
    args: argparse.Namespace, seed: int, h: float, gamma0: np.ndarray
) -> dict[str, Any]:
    return {
        "model": "random_ising_antiferromagnet",
        "scan_parameter": "Gamma0",
        "L": int(args.L),
        "seed": int(seed),
        "h": float(h),
        "ensemble": args.ensemble,
        "bc": normalize_bc(getattr(args, "bc", "open")),
        "chi_max": int(args.chi),
        "Gamma0_values": gamma0.tolist(),
        "initial_states": list(args.initial_states),
        "multistart_random_seed": int(args.random_seed),
        "two_site_sweeps": int(args.two_site_sweeps),
        "one_site_sweeps": int(args.one_site_sweeps),
        "max_E_err": float(args.max_E_err),
        "max_S_err": float(args.max_S_err),
        "svd_min": float(args.svd_min),
        "mixer": not args.no_mixer,
        "branch_energy_tolerance": float(args.branch_energy_tol),
        "branch_infidelity_tolerance": float(args.branch_infidelity_tol),
        "selection_energy_tolerance": float(args.selection_energy_tol),
        "mps_save_policy": args.save_mps,
    }


def metadata_compatibility(
    metadata: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    exact_keys = (
        "model",
        "scan_parameter",
        "L",
        "seed",
        "ensemble",
        "bc",
        "chi_max",
        "initial_states",
        "multistart_random_seed",
        "two_site_sweeps",
        "one_site_sweeps",
        "mixer",
    )
    float_keys = (
        "h",
        "max_E_err",
        "max_S_err",
        "svd_min",
        "branch_energy_tolerance",
        "branch_infidelity_tolerance",
        "selection_energy_tolerance",
    )
    for key in exact_keys:
        if metadata.get(key) != expected[key]:
            reasons.append(key)
    if metadata.get("mps_save_policy", "none") != expected["mps_save_policy"]:
        reasons.append("mps_save_policy")
    for key in float_keys:
        if not _float_matches(metadata.get(key), expected[key]):
            reasons.append(key)
    if not grid_contains(metadata.get("Gamma0_values"), np.asarray(expected["Gamma0_values"])):
        reasons.append("Gamma0_values")
    return not reasons, reasons


def validate_summary_directory(
    directory: Path, expected: dict[str, Any]
) -> tuple[bool, str, int]:
    npz_path = directory / SUMMARY_NPZ
    json_path = directory / SUMMARY_JSON
    if not npz_path.exists() and not json_path.exists():
        return False, "missing", 0
    if not npz_path.exists() or not json_path.exists():
        return False, "partial result (one summary file is missing)", 0
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload["metadata"]
        compatible, reasons = metadata_compatibility(metadata, expected)
        if not compatible:
            return False, "incompatible metadata: " + ", ".join(reasons), 0
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = sorted(REQUIRED_ARRAYS.difference(archive.files))
            if missing:
                return False, "missing arrays: " + ", ".join(missing), 0
            gamma0 = np.asarray(archive["Gamma0_values"], dtype=float)
            L = int(expected["L"])
            G = gamma0.size
            if (
                gamma0.ndim != 1
                or G == 0
                or not np.all(np.isfinite(gamma0))
                or np.any(np.diff(gamma0) <= 0.0)
            ):
                return False, "invalid NPZ Gamma0 grid", 0
            json_gamma0 = np.asarray(metadata.get("Gamma0_values"), dtype=float)
            if not np.array_equal(gamma0, json_gamma0):
                return False, "JSON and NPZ Gamma0 grids differ", 0
            expected_shapes = {
                "Gamma": (G, L),
                "Czz": (G, L, L),
                "Cstag": (G, L, L),
                "D": (G, L, L),
            }
            for key, shape in expected_shapes.items():
                if archive[key].shape != shape:
                    return False, f"invalid {key} shape {archive[key].shape}, expected {shape}", 0
            one_dimensional = (
                "forward_energy", "backward_energy", "absolute_energy_difference",
                "branch_fidelity", "branch_disagreement",
                "forward_max_bond_dimension", "backward_max_bond_dimension",
                "selected_branch", "half_chain_entropy",
            )
            for key in one_dimensional:
                if archive[key].shape != (G,):
                    return False, f"invalid {key} shape {archive[key].shape}", 0
            n_bonds = L - 1 if expected["bc"] == "open" else L
            if archive["u"].shape != (n_bonds,) or archive["J"].shape != (n_bonds,):
                return False, f"invalid {expected['bc']} bond-disorder shapes", 0
            if archive["v"].shape != (L,):
                return False, "invalid field-disorder shape", 0
            finite_keys = (
                "u", "v", "J", "Gamma", "forward_energy", "backward_energy",
                "absolute_energy_difference", "branch_fidelity",
                "half_chain_entropy", "Czz", "Cstag", "D",
            )
            for key in finite_keys:
                if not np.all(np.isfinite(archive[key])):
                    return False, f"non-finite values in {key}", 0
            labels = archive["selected_branch"].astype(str)
            if np.any(~np.isin(labels, ("forward", "backward"))):
                return False, "invalid selected_branch label", 0
            if len(payload.get("rows", [])) != G:
                return False, "JSON row count differs from the saved grid", 0
            czz = np.asarray(archive["Czz"], dtype=float)
            cstag = np.asarray(archive["Cstag"], dtype=float)
            distance = np.asarray(archive["D"], dtype=float)
            parity = (-1.0) ** np.abs(
                np.arange(L)[:, None] - np.arange(L)[None, :]
            )
            if not np.allclose(cstag, czz * parity, rtol=0.0, atol=1e-9):
                return False, "Cstag convention check failed", 0
            if not np.allclose(distance, 1.0 - cstag, rtol=0.0, atol=1e-9):
                return False, "D=1-Cstag check failed", 0
            if not grid_contains(gamma0, np.asarray(expected["Gamma0_values"])):
                return False, "NPZ Gamma0 grid does not contain requested grid", 0
            if not _float_matches(np.asarray(archive["h"]).item(), expected["h"]):
                return False, "NPZ h is inconsistent", 0
            mps_path = directory / MPS_COLLECTION_FILE
            mps_manifest_path = directory / MPS_MANIFEST_FILE
            if expected["mps_save_policy"] == "selected":
                if not mps_path.exists() or not mps_manifest_path.exists():
                    return False, "selected MPS collection or manifest is missing", 0
                with mps_manifest_path.open("r", encoding="utf-8") as handle:
                    mps_manifest = json.load(handle)
                manifest_gamma0 = np.asarray(
                    mps_manifest.get("Gamma0_values"), dtype=float
                )
                if not np.array_equal(manifest_gamma0, gamma0):
                    return False, "MPS manifest Gamma0 grid differs", 0
                manifest_exact = {
                    "model": "random_ising_antiferromagnet",
                    "scan_parameter": "Gamma0",
                    "L": expected["L"],
                    "seed": expected["seed"],
                    "ensemble": expected["ensemble"],
                    "bc": expected["bc"],
                    "chi_max": expected["chi_max"],
                }
                for key, value in manifest_exact.items():
                    if mps_manifest.get(key) != value:
                        return False, f"MPS manifest has inconsistent {key}", 0
                if not _float_matches(mps_manifest.get("h"), expected["h"]):
                    return False, "MPS manifest has inconsistent h", 0
                point_records = mps_manifest.get("point_records", [])
                if len(point_records) != G:
                    return False, "MPS manifest point count differs", 0
                state_keys = [record.get("state_key") for record in point_records]
                if len(set(state_keys)) != G or any(not key for key in state_keys):
                    return False, "MPS manifest has invalid state keys", 0
                selected_labels = archive["selected_branch"].astype(str)
                disagreement_flags = archive["branch_disagreement"].astype(bool)
                for index, record in enumerate(point_records):
                    if record.get("point_index") != index:
                        return False, "MPS manifest has invalid point indices", 0
                    if not _float_matches(record.get("Gamma0"), gamma0[index]):
                        return False, "MPS manifest has inconsistent Gamma0 records", 0
                    if record.get("selected_branch") != selected_labels[index]:
                        return False, "MPS manifest has inconsistent selected branches", 0
                    if bool(record.get("branch_disagreement")) != disagreement_flags[index]:
                        return False, "MPS manifest has inconsistent disagreement flags", 0
                collection_info = inspect_mps_collection_hdf5(mps_path)
                if collection_info["selected_state_count"] != G:
                    return False, "MPS collection selected-state count differs", 0
                disagreement_count = int(np.sum(archive["branch_disagreement"]))
                if collection_info["disagreement_point_count"] != disagreement_count:
                    return False, "MPS collection disagreement count differs", 0
            elif mps_path.exists() or mps_manifest_path.exists():
                return False, "unexpected MPS artifacts for save policy none", 0
            exact_grid = np.array_equal(
                gamma0, np.asarray(expected["Gamma0_values"], dtype=float)
            )
            if not exact_grid:
                requested_indices = np.asarray([
                    int(np.argmin(np.abs(gamma0 - value)))
                    for value in expected["Gamma0_values"]
                ])
                if np.any(archive["branch_disagreement"][requested_indices].astype(bool)):
                    return False, "grid superset has branch disagreement at requested points", 0
                if metadata.get("forward_seed_failures") or metadata.get("backward_seed_failures"):
                    return False, "grid superset has endpoint multistart failures", 0
                return True, "compatible grid superset; continuation protocol differs", int(G)
        return True, "compatible exact grid", int(G)
    except (OSError, ValueError, KeyError, ImportError, json.JSONDecodeError) as exc:
        return False, f"unreadable summary: {exc}", 0


def compatible_external_scan(
    roots: list[Path], expected: dict[str, Any], target: Path
) -> Path | None:
    candidates: list[tuple[int, str, Path]] = []
    requested_count = len(expected["Gamma0_values"])
    for root in roots:
        if not root.exists():
            continue
        for json_path in root.rglob(SUMMARY_JSON):
            directory = json_path.parent
            if directory.resolve() == target.resolve():
                continue
            valid, _message, saved_count = validate_summary_directory(directory, expected)
            if valid:
                candidates.append((saved_count - requested_count, str(directory), directory))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def build_scan_command(
    args: argparse.Namespace,
    seed: int,
    h: float,
    gamma0: np.ndarray,
    scan_output_root: Path,
    directory_name: str | None = None,
    overwrite: bool = False,
) -> list[str]:
    if directory_name is None:
        directory_name = scan_directory_name(
            args.L, seed, h, args.chi, args.campaign_name
        )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_random_ising_gamma_scan.py"),
        "--L", str(args.L),
        "--seed", str(seed),
        "--h", format(h, ".17g"),
        "--ensemble", args.ensemble,
        "--bc", normalize_bc(getattr(args, "bc", "open")),
        "--Gamma0", *[format(value, ".17g") for value in gamma0],
        "--chi", str(args.chi),
        "--initial-states", *list(args.initial_states),
        "--random-seed", str(args.random_seed),
        "--two-site-sweeps", str(args.two_site_sweeps),
        "--one-site-sweeps", str(args.one_site_sweeps),
        "--max-E-err", format(args.max_E_err, ".17g"),
        "--max-S-err", format(args.max_S_err, ".17g"),
        "--svd-min", format(args.svd_min, ".17g"),
        "--branch-energy-tol", format(args.branch_energy_tol, ".17g"),
        "--branch-infidelity-tol", format(args.branch_infidelity_tol, ".17g"),
        "--selection-energy-tol", format(args.selection_energy_tol, ".17g"),
        "--save-mps", args.save_mps,
        "--run-name", args.campaign_name,
        "--directory-name", directory_name,
        "--output-root", str(scan_output_root),
    ]
    if args.no_mixer:
        command.append("--no-mixer")
    if overwrite:
        command.append("--overwrite")
    return command


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON file, tolerating transient Windows locks."""

    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, payload)
    replace_with_retry(temporary, path)


def campaign_config(
    args: argparse.Namespace,
    seeds: np.ndarray,
    h_values: np.ndarray,
    gamma0: np.ndarray,
    scan_output_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_name": args.campaign_name,
        "worker": "scripts/run_random_ising_gamma_scan.py",
        "L": int(args.L),
        "seeds": seeds.tolist(),
        "h_values": h_values.tolist(),
        "Gamma0_values": gamma0.tolist(),
        "ensemble": args.ensemble,
        "bc": normalize_bc(getattr(args, "bc", "open")),
        "chi_max": int(args.chi),
        "initial_states": list(args.initial_states),
        "multistart_random_seed": int(args.random_seed),
        "two_site_sweeps": int(args.two_site_sweeps),
        "one_site_sweeps": int(args.one_site_sweeps),
        "max_E_err": float(args.max_E_err),
        "max_S_err": float(args.max_S_err),
        "svd_min": float(args.svd_min),
        "mixer": not args.no_mixer,
        "branch_energy_tolerance": float(args.branch_energy_tol),
        "branch_infidelity_tolerance": float(args.branch_infidelity_tol),
        "selection_energy_tolerance": float(args.selection_energy_tol),
        "mps_save_policy": args.save_mps,
        # Keep an explicitly supplied short drive mapping (e.g. ``Q:``) in
        # the command path.  ``resolve()`` expands a Windows SUBST drive back
        # to a long OneDrive path and can reintroduce MAX_PATH failures while
        # saving selected MPS artifacts.
        "scan_output_root": str(scan_output_root.absolute()),
    }


def validate_args(args: argparse.Namespace) -> None:
    validate_campaign_name(args.campaign_name)
    normalize_bc(getattr(args, "bc", "open"))
    if args.L < 4:
        raise ValueError("ensemble analysis requires L at least 4")
    if args.chi < 1:
        raise ValueError("chi must be positive")
    if args.pairs_file is not None:
        if args.h is not None or args.seeds is not None:
            raise ValueError("--pairs-file cannot be combined with --h or --seeds")
    elif args.h is None:
        raise ValueError("--h is required unless --pairs-file is used")
    if args.random_seed < 0:
        raise ValueError("random-seed must be non-negative")
    if args.two_site_sweeps < 1 or args.one_site_sweeps < 0:
        raise ValueError("invalid DMRG sweep counts")
    tolerances = (
        args.max_E_err, args.max_S_err, args.svd_min,
        args.branch_energy_tol, args.branch_infidelity_tol,
        args.selection_energy_tol,
    )
    if not np.all(np.isfinite(tolerances)) or np.any(np.asarray(tolerances) < 0.0):
        raise ValueError("DMRG and branch tolerances must be finite and non-negative")
    known_states = {
        "classical_ground_state", "neel_up", "neel_down",
        "all_up", "all_down", "random_z",
    }
    if not args.initial_states or len(set(args.initial_states)) != len(args.initial_states):
        raise ValueError("initial-states must be non-empty and unique")
    unknown = sorted(set(args.initial_states).difference(known_states))
    if unknown:
        raise ValueError("unknown initial-state labels: " + ", ".join(unknown))


def main() -> None:
    args = parse_args()
    validate_args(args)
    pair_grid = load_pair_grid(args.pairs_file) if args.pairs_file is not None else None
    if pair_grid is None:
        seeds = build_seed_grid(args)
        h_values = build_h_grid(args)
        job_grid = [
            (int(seed), float(h)) for h in h_values for seed in seeds
        ]
    else:
        seeds = np.asarray(sorted({seed for seed, _h in pair_grid}), dtype=int)
        h_values = np.asarray(sorted({h for _seed, h in pair_grid}), dtype=float)
        job_grid = pair_grid
    gamma0 = build_gamma0_grid(args)

    # Do not resolve a SUBST drive: the compact path is operationally
    # significant on Windows installations without long-path support.
    output_root = args.output_root.absolute()
    campaign_dir = output_root / args.campaign_name
    scan_output_root = campaign_dir / "scans"

    targets: set[Path] = set()
    jobs: list[dict[str, Any]] = []
    for seed, h in job_grid:
        compact_target = scan_output_root / scan_directory_name(
            args.L, int(seed), float(h), args.chi, args.campaign_name
        )
        legacy_target = scan_output_root / legacy_scan_directory_name(
            args.L, int(seed), float(h), args.chi, args.campaign_name
        )
        target = (
            legacy_target
            if scan_directory_has_artifacts(legacy_target)
            else compact_target
        )
        # Preserve a short explicitly supplied Windows drive mapping.  Resolving
        # a SUBST drive expands it to the OneDrive path and makes post-run
        # validation hit MAX_PATH even when the scan itself saved correctly.
        resolved = target.absolute()
        if resolved in targets:
            raise ValueError(
                "distinct jobs collide after formatting h with 12 significant digits"
            )
        targets.add(resolved)
        jobs.append({
            "h": float(h),
            "seed": int(seed),
            "target_directory": str(resolved),
            "status": "pending",
        })

    campaign_dir.mkdir(parents=True, exist_ok=True)
    scan_output_root.mkdir(parents=True, exist_ok=True)
    config_path = campaign_dir / "campaign_config.json"
    status_path = campaign_dir / "campaign_status.json"
    config = campaign_config(args, seeds, h_values, gamma0, scan_output_root)
    if pair_grid is not None:
        config["campaign_kind"] = "directed_seed_h_pairs"
        config["job_pairs"] = [
            {"seed": int(seed), "h": float(h)} for seed, h in pair_grid
        ]
        config["pairs_file"] = str(args.pairs_file.resolve())
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            previous_config = json.load(handle)
        previous_config.setdefault("mps_save_policy", "none")
        if previous_config != config:
            raise ValueError(
                f"campaign {args.campaign_name!r} already exists with a different "
                "configuration; choose a new campaign-name"
            )
    else:
        atomic_write_json(config_path, config)

    status: dict[str, Any] = {
        "schema_version": 1,
        "campaign_name": args.campaign_name,
        "config_path": str(config_path.resolve()),
        "created_or_resumed_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs,
    }
    atomic_write_json(status_path, status)
    reuse_roots = [path.resolve() for path in args.reuse_compatible_from]

    failures = 0
    for index, job in enumerate(jobs, start=1):
        seed = int(job["seed"])
        h = float(job["h"])
        target = Path(job["target_directory"])
        expected = expected_metadata(args, seed, h, gamma0)
        valid, message, _saved_count = validate_summary_directory(target, expected)
        overwrite = False
        if valid:
            job["status"] = "complete"
            job["source_directory"] = str(target)
            job["message"] = message
            job["continuation_protocol_matches"] = True
            print(f"[{index}/{len(jobs)}] SKIP seed={seed}, h={h:g}: complete")
            atomic_write_json(status_path, status)
            continue

        target_has_files = scan_directory_has_artifacts(target)
        if target_has_files:
            both_summaries = (
                (target / SUMMARY_NPZ).exists() and (target / SUMMARY_JSON).exists()
            )
            if both_summaries and message.startswith("incompatible metadata"):
                raise RuntimeError(
                    f"target {target} is a complete result with incompatible metadata "
                    f"({message}); choose a new campaign-name"
                )
            if not args.rerun_incomplete:
                raise RuntimeError(
                    f"target {target} is not reusable: {message}; pass "
                    "--rerun-incomplete to recompute it"
                )
            overwrite = True
        elif reuse_roots:
            reused = compatible_external_scan(reuse_roots, expected, target)
            if reused is not None:
                _valid, reuse_message, saved_count = validate_summary_directory(
                    reused, expected
                )
                job["status"] = "reused"
                job["source_directory"] = str(reused.resolve())
                job["message"] = reuse_message
                job["source_grid_points"] = saved_count
                job["continuation_protocol_matches"] = saved_count == gamma0.size
                print(
                    f"[{index}/{len(jobs)}] REUSE seed={seed}, h={h:g}: "
                    f"{reused}"
                )
                atomic_write_json(status_path, status)
                continue

        command = build_scan_command(
            args,
            seed,
            h,
            gamma0,
            scan_output_root,
            directory_name=target.name,
            overwrite=overwrite,
        )
        job["command"] = command
        if args.dry_run:
            job["status"] = "planned"
            job["message"] = message
            print(f"[{index}/{len(jobs)}] PLAN seed={seed}, h={h:g}")
            print(subprocess.list2cmdline(command))
            continue

        job["status"] = "running"
        job["started_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(status_path, status)
        print(f"[{index}/{len(jobs)}] RUN seed={seed}, h={h:g}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        job["return_code"] = int(completed.returncode)
        job["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if completed.returncode == 0:
            valid, validation_message, _saved_count = validate_summary_directory(
                target, expected
            )
            if valid:
                job["status"] = "complete"
                job["source_directory"] = str(target)
                job["message"] = validation_message
                job["continuation_protocol_matches"] = True
            else:
                job["status"] = "failed"
                job["message"] = validation_message
                failures += 1
        else:
            job["status"] = "failed"
            job["message"] = f"worker exited with code {completed.returncode}"
            failures += 1
        atomic_write_json(status_path, status)
        if job["status"] == "failed" and not args.continue_on_error:
            raise RuntimeError(job["message"])

    counts: dict[str, int] = {}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    status["counts"] = counts
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(status_path, status)
    print(f"Campaign status written to: {status_path}")
    print("Status counts: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if failures:
        raise RuntimeError(f"campaign finished with {failures} failed job(s)")


if __name__ == "__main__":
    main()
