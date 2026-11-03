from __future__ import annotations

import argparse
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

from clean_tfim_tn.io.clean_ising_gamma_scan import (  # noqa: E402
    MPS_COLLECTION_FILE, MPS_MANIFEST_FILE, SUMMARY_JSON, SUMMARY_NPZ,
    atomic_write_json, load_json, validate_clean_ising_gamma_scan,
)
from clean_tfim_tn.tn.clean_ising_gamma_continuation import (  # noqa: E402
    DEFAULT_CLEAN_INITIAL_STATES,
)
from clean_tfim_tn.disorder import normalize_bc  # noqa: E402


NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable clean Ising campaign over (L,h,Gamma)."
    )
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--L", type=int, nargs="+", required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--bc", choices=("open", "obc", "periodic", "pbc"), default="open")
    parser.add_argument("--h", type=float, nargs="+", default=None)
    parser.add_argument("--h-min", type=float, default=0.0)
    parser.add_argument("--h-max", type=float, default=2.5)
    parser.add_argument("--h-points", type=int, default=26)
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
    parser.add_argument("--rerun-incomplete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/clean_ising_gamma_campaign")
    )
    return parser.parse_args()


def grid(explicit: Any, lower: float, upper: float, points: int, name: str) -> np.ndarray:
    if explicit is None:
        if points < 1 or upper < lower:
            raise ValueError(f"invalid {name} interval or point count")
        values = np.linspace(lower, upper, points)
    else:
        values = np.asarray(explicit, dtype=float)
    if values.size < 1 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} values must be non-empty, finite and non-negative")
    return np.unique(values)


def validate_args(args: argparse.Namespace) -> None:
    if not NAME_PATTERN.fullmatch(args.campaign_name):
        raise ValueError("campaign-name contains unsupported characters")
    if any(L < 2 or L % 2 for L in args.L):
        raise ValueError("all L values must be even integers at least 2")
    if not np.isfinite(args.J) or args.J <= 0.0:
        raise ValueError("J must be finite and positive")
    if args.chi < 1 or args.two_site_sweeps < 1 or args.one_site_sweeps < 0:
        raise ValueError("invalid chi or sweep counts")
    if not args.initial_states or len(set(args.initial_states)) != len(args.initial_states):
        raise ValueError("initial-states must be non-empty and unique")
    unknown = set(args.initial_states).difference(DEFAULT_CLEAN_INITIAL_STATES)
    if unknown:
        raise ValueError(f"unknown deterministic initial states: {sorted(unknown)}")
    tolerances = np.asarray([
        args.max_E_err, args.max_S_err, args.svd_min,
        args.branch_energy_tol, args.branch_infidelity_tol,
        args.selection_energy_tol,
    ])
    if np.any(~np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("DMRG and branch tolerances must be finite and non-negative")


def campaign_config(
    args: argparse.Namespace, h_values: np.ndarray, gamma_values: np.ndarray
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_name": args.campaign_name,
        "model": "clean_mixed_field_ising",
        "disorder": "none",
        "worker": "scripts/run_clean_ising_gamma_scan.py",
        "L_values": sorted(set(args.L)),
        "J": float(args.J),
        "h_values": h_values.tolist(),
        "Gamma_values": gamma_values.tolist(),
        "bc": normalize_bc(args.bc),
        "conserve": None,
        "chi_max": int(args.chi),
        "initial_states": list(args.initial_states),
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


def expected(config: dict[str, Any], L: int, h: float) -> dict[str, Any]:
    keys = (
        "model", "disorder", "J", "bc", "conserve", "chi_max",
        "initial_states", "two_site_sweeps", "one_site_sweeps", "max_E_err",
        "max_S_err", "svd_min", "mixer", "branch_energy_tolerance",
        "branch_infidelity_tolerance", "selection_energy_tolerance", "mps_save_policy",
    )
    result = {key: config[key] for key in keys}
    result.update({"scan_parameter": "Gamma", "L": L, "h": h, "run_name": config["campaign_name"]})
    return result


def scan_directory(campaign: Path, config: dict[str, Any], L: int, h: float) -> Path:
    return campaign / "scans" / scan_directory_name(config, L, h)


def scan_directory_name(config: dict[str, Any], L: int, h: float) -> str:
    # The campaign directory already contains campaign_name. Repeating it here
    # made continuation_summary.tmp cross MAX_PATH under Windows/OneDrive.
    return f"L{L}_h_{h:.12g}_J{config['J']:.12g}_chi{config['chi_max']}"


def worker_command(
    config: dict[str, Any], L: int, h: float, campaign: Path, overwrite: bool
) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts" / "run_clean_ising_gamma_scan.py"),
        "--L", str(L), "--J", format(config["J"], ".17g"),
        "--bc", config["bc"],
        "--h", format(h, ".17g"), "--Gamma",
        *[format(value, ".17g") for value in config["Gamma_values"]],
        "--chi", str(config["chi_max"]), "--initial-states", *config["initial_states"],
        "--two-site-sweeps", str(config["two_site_sweeps"]),
        "--one-site-sweeps", str(config["one_site_sweeps"]),
        "--max-E-err", format(config["max_E_err"], ".17g"),
        "--max-S-err", format(config["max_S_err"], ".17g"),
        "--svd-min", format(config["svd_min"], ".17g"),
        "--branch-energy-tol", format(config["branch_energy_tolerance"], ".17g"),
        "--branch-infidelity-tol", format(config["branch_infidelity_tolerance"], ".17g"),
        "--selection-energy-tol", format(config["selection_energy_tolerance"], ".17g"),
        "--save-mps", config["mps_save_policy"], "--run-name", config["campaign_name"],
        "--directory-name", scan_directory_name(config, L, h),
        "--output-root", str((campaign / "scans").resolve()),
    ]
    if not config["mixer"]:
        command.append("--no-mixer")
    if overwrite:
        command.append("--overwrite")
    return command


def has_any_artifact(directory: Path) -> bool:
    return any((directory / name).exists() for name in (
        SUMMARY_NPZ, SUMMARY_JSON, MPS_COLLECTION_FILE, MPS_MANIFEST_FILE
    ))


def main() -> None:
    args = parse_args()
    validate_args(args)
    h_values = grid(args.h, args.h_min, args.h_max, args.h_points, "h")
    gamma_values = grid(
        args.Gamma, args.Gamma_min, args.Gamma_max, args.Gamma_points, "Gamma"
    )
    config = campaign_config(args, h_values, gamma_values)
    campaign = args.output_root.resolve() / args.campaign_name
    campaign.mkdir(parents=True, exist_ok=True)
    config_path, status_path = campaign / "campaign_config.json", campaign / "campaign_status.json"
    if config_path.exists():
        if load_json(config_path) != config:
            raise ValueError("campaign configuration differs from the immutable saved config")
    else:
        atomic_write_json(config_path, config)

    jobs = []
    targets: set[Path] = set()
    for L in config["L_values"]:
        for h in h_values:
            directory = scan_directory(campaign, config, int(L), float(h))
            resolved = directory.resolve()
            longest_artifact = max(
                (resolved / name for name in (
                    SUMMARY_NPZ, SUMMARY_JSON, MPS_COLLECTION_FILE, MPS_MANIFEST_FILE,
                    "continuation_summary.tmp", "selected_mps_manifest.tmp",
                )),
                key=lambda path: len(str(path)),
            )
            if sys.platform == "win32" and len(str(longest_artifact)) >= 260:
                raise ValueError(
                    f"target path is too long for Windows ({len(str(longest_artifact))} "
                    f"characters): {longest_artifact}; choose a shorter campaign-name "
                    "or output-root"
                )
            if resolved in targets:
                raise ValueError("distinct jobs collide after formatting h")
            targets.add(resolved)
            complete, message = False, "missing"
            try:
                scan = validate_clean_ising_gamma_scan(
                    directory, expected=expected(config, int(L), float(h)),
                    require_mps=config["mps_save_policy"] == "selected",
                )
                if not np.array_equal(scan["arrays"]["Gamma_values"], gamma_values):
                    raise ValueError("saved Gamma grid differs from campaign config")
                complete, message = True, "compatible exact grid"
            except (FileNotFoundError, ValueError, OSError, EOFError) as exc:
                if directory.exists():
                    message = f"incomplete or incompatible: {exc}"
            overwrite = has_any_artifact(directory) and not complete
            if overwrite and not args.rerun_incomplete:
                raise RuntimeError(
                    f"target {directory} is not reusable ({message}); pass --rerun-incomplete"
                )
            job = {
                "L": int(L), "h": float(h),
                "status": "complete" if complete else "planned",
                "message": message, "target_directory": str(resolved),
            }
            if complete:
                job["source_directory"] = str(resolved)
            else:
                job["command"] = worker_command(config, int(L), float(h), campaign, overwrite)
            jobs.append(job)

    status = {
        "schema_version": 1, "campaign_name": args.campaign_name,
        "created_or_resumed_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()), "jobs": jobs,
    }
    atomic_write_json(status_path, status)
    print(
        f"Campaign plan: complete={sum(x['status']=='complete' for x in jobs)}, "
        f"planned={sum(x['status']=='planned' for x in jobs)}"
    )
    if args.dry_run:
        for job in jobs:
            print(f"[{job['status']}] L={job['L']}, h={job['h']:g}: {job['message']}")
            if "command" in job:
                print(subprocess.list2cmdline(job["command"]))
        return

    failures = 0
    for index, job in enumerate(jobs, start=1):
        if job["status"] == "complete":
            print(f"[{index}/{len(jobs)}] SKIP L={job['L']}, h={job['h']:g}")
            continue
        print(f"[{index}/{len(jobs)}] RUN L={job['L']}, h={job['h']:g}", flush=True)
        job["status"] = "running"
        job["started_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(status_path, status)
        completed = subprocess.run(job["command"], cwd=ROOT, check=False)
        job["return_code"] = int(completed.returncode)
        job["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if completed.returncode != 0:
            job["status"] = "failed"
            job["message"] = f"worker exited with code {completed.returncode}"
            failures += 1
        else:
            directory = Path(job["target_directory"])
            try:
                validate_clean_ising_gamma_scan(
                    directory, expected=expected(config, int(job["L"]), float(job["h"])),
                    require_mps=config["mps_save_policy"] == "selected",
                )
                job["status"] = "complete"
                job["message"] = "compatible exact grid"
                job["source_directory"] = str(directory.resolve())
            except (FileNotFoundError, ValueError, OSError, EOFError) as exc:
                job["status"] = "failed"
                job["message"] = str(exc)
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
    if failures:
        raise RuntimeError(f"campaign finished with {failures} failed job(s)")
    print(f"Campaign complete: {campaign}")


if __name__ == "__main__":
    main()
