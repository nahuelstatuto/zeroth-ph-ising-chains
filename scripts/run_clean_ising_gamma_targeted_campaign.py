from __future__ import annotations

import argparse
import json
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
    MPS_COLLECTION_FILE,
    MPS_MANIFEST_FILE,
    SUMMARY_JSON,
    SUMMARY_NPZ,
    atomic_write_json,
    load_json,
    validate_clean_ising_gamma_scan,
)
from clean_tfim_tn.tn.clean_ising_gamma_continuation import (  # noqa: E402
    DEFAULT_CLEAN_INITIAL_STATES,
)


NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable clean Ising campaign with an independent Gamma grid "
            "for every h."
        )
    )
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--boundary-spec", type=Path, required=True)
    parser.add_argument("--L", type=int, nargs="+", required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument(
        "--initial-states", nargs="+", default=list(DEFAULT_CLEAN_INITIAL_STATES)
    )
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


def validate_args(args: argparse.Namespace) -> None:
    if not NAME_PATTERN.fullmatch(args.campaign_name):
        raise ValueError("campaign-name contains unsupported characters")
    if any(L < 2 for L in args.L):
        raise ValueError("all L values must be integers at least 2")
    if not np.isfinite(args.J) or args.J <= 0.0:
        raise ValueError("J must be finite and positive")
    if args.chi < 1 or args.two_site_sweeps < 1 or args.one_site_sweeps < 0:
        raise ValueError("invalid chi or sweep counts")
    if not args.initial_states or len(set(args.initial_states)) != len(args.initial_states):
        raise ValueError("initial-states must be non-empty and unique")
    unknown = set(args.initial_states).difference(DEFAULT_CLEAN_INITIAL_STATES)
    if unknown:
        raise ValueError(f"unknown deterministic initial states: {sorted(unknown)}")
    tolerances = np.asarray(
        [
            args.max_E_err,
            args.max_S_err,
            args.svd_min,
            args.branch_energy_tol,
            args.branch_infidelity_tol,
            args.selection_energy_tol,
        ]
    )
    if np.any(~np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("DMRG and branch tolerances must be finite and non-negative")


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def load_boundary_points(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise ValueError("boundary spec must be a JSON object")
    raw_points = spec.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("boundary spec must contain a non-empty points list")

    raw_offsets = spec.get("Gamma_offsets")
    offsets: np.ndarray | None = None
    if raw_offsets is not None:
        offsets = np.asarray(raw_offsets, dtype=float)
        if (
            offsets.ndim != 1
            or offsets.size < 1
            or np.any(~np.isfinite(offsets))
        ):
            raise ValueError("Gamma_offsets must be a non-empty list of finite numbers")
        offsets = np.unique(offsets)

    points: list[dict[str, Any]] = []
    seen_h: set[float] = set()
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, dict):
            raise ValueError(f"points[{index}] must be an object")
        h = _finite_scalar(raw.get("h"), f"points[{index}].h")
        if h < 0.0:
            raise ValueError(f"points[{index}].h must be non-negative")
        if h in seen_h:
            raise ValueError(f"duplicate h value in boundary spec: {h:g}")
        seen_h.add(h)

        center_value = raw.get("Gamma_center")
        center = (
            None
            if center_value is None
            else _finite_scalar(center_value, f"points[{index}].Gamma_center")
        )
        if center is not None and center < 0.0:
            raise ValueError(f"points[{index}].Gamma_center must be non-negative")

        explicit = raw.get("Gamma_values")
        if explicit is not None:
            gamma = np.asarray(explicit, dtype=float)
            if (
                gamma.ndim != 1
                or gamma.size < 1
                or np.any(~np.isfinite(gamma))
                or np.any(gamma < 0.0)
            ):
                raise ValueError(
                    f"points[{index}].Gamma_values must be finite and non-negative"
                )
            gamma = np.unique(gamma)
        else:
            if center is None or offsets is None:
                raise ValueError(
                    f"points[{index}] needs Gamma_values or Gamma_center with "
                    "top-level Gamma_offsets"
                )
            gamma = np.unique(np.maximum(0.0, center + offsets))

        points.append(
            {
                "h": h,
                "Gamma_center": center,
                "Gamma_values": gamma.tolist(),
            }
        )

    points.sort(key=lambda item: item["h"])
    normalized_spec = {
        "schema_version": int(spec.get("schema_version", 1)),
        "description": str(spec.get("description", "")),
        "source": str(spec.get("source", "")),
        "Gamma_offsets": None if offsets is None else offsets.tolist(),
        "points": points,
    }
    return normalized_spec, points


def campaign_config(
    args: argparse.Namespace,
    boundary_spec: dict[str, Any],
    points: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "campaign_kind": "targeted_boundary",
        "campaign_name": args.campaign_name,
        "model": "clean_mixed_field_ising",
        "disorder": "none",
        "worker": "scripts/run_clean_ising_gamma_scan.py",
        "launcher": "scripts/run_clean_ising_gamma_targeted_campaign.py",
        "L_values": sorted(set(args.L)),
        "J": float(args.J),
        "h_values": [point["h"] for point in points],
        "boundary_spec": boundary_spec,
        "Gamma_values_by_h": points,
        "bc": "open",
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
        "model",
        "disorder",
        "J",
        "bc",
        "conserve",
        "chi_max",
        "initial_states",
        "two_site_sweeps",
        "one_site_sweeps",
        "max_E_err",
        "max_S_err",
        "svd_min",
        "mixer",
        "branch_energy_tolerance",
        "branch_infidelity_tolerance",
        "selection_energy_tolerance",
        "mps_save_policy",
    )
    result = {key: config[key] for key in keys}
    result.update(
        {
            "scan_parameter": "Gamma",
            "L": L,
            "h": h,
            "run_name": config["campaign_name"],
        }
    )
    return result


def scan_directory_name(config: dict[str, Any], L: int, h: float) -> str:
    return f"L{L}_h_{h:.12g}_J{config['J']:.12g}_chi{config['chi_max']}"


def worker_command(
    config: dict[str, Any],
    L: int,
    point: dict[str, Any],
    campaign: Path,
    overwrite: bool,
) -> list[str]:
    h = float(point["h"])
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_clean_ising_gamma_scan.py"),
        "--L",
        str(L),
        "--J",
        format(config["J"], ".17g"),
        "--h",
        format(h, ".17g"),
        "--Gamma",
        *[format(value, ".17g") for value in point["Gamma_values"]],
        "--chi",
        str(config["chi_max"]),
        "--initial-states",
        *config["initial_states"],
        "--two-site-sweeps",
        str(config["two_site_sweeps"]),
        "--one-site-sweeps",
        str(config["one_site_sweeps"]),
        "--max-E-err",
        format(config["max_E_err"], ".17g"),
        "--max-S-err",
        format(config["max_S_err"], ".17g"),
        "--svd-min",
        format(config["svd_min"], ".17g"),
        "--branch-energy-tol",
        format(config["branch_energy_tolerance"], ".17g"),
        "--branch-infidelity-tol",
        format(config["branch_infidelity_tolerance"], ".17g"),
        "--selection-energy-tol",
        format(config["selection_energy_tolerance"], ".17g"),
        "--save-mps",
        config["mps_save_policy"],
        "--run-name",
        config["campaign_name"],
        "--directory-name",
        scan_directory_name(config, L, h),
        "--output-root",
        str((campaign / "scans").resolve()),
    ]
    if not config["mixer"]:
        command.append("--no-mixer")
    if overwrite:
        command.append("--overwrite")
    return command


def has_any_artifact(directory: Path) -> bool:
    return any(
        (directory / name).exists()
        for name in (SUMMARY_NPZ, SUMMARY_JSON, MPS_COLLECTION_FILE, MPS_MANIFEST_FILE)
    )


def validate_path_length(directory: Path) -> None:
    resolved = directory.resolve()
    longest_artifact = max(
        (
            resolved / name
            for name in (
                SUMMARY_NPZ,
                SUMMARY_JSON,
                MPS_COLLECTION_FILE,
                MPS_MANIFEST_FILE,
                "continuation_summary.tmp",
                "selected_mps_manifest.tmp",
            )
        ),
        key=lambda item: len(str(item)),
    )
    if sys.platform == "win32" and len(str(longest_artifact)) >= 260:
        raise ValueError(
            f"target path is too long for Windows ({len(str(longest_artifact))} "
            f"characters): {longest_artifact}; choose a shorter campaign-name "
            "or output-root"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    boundary_spec, points = load_boundary_points(args.boundary_spec)
    config = campaign_config(args, boundary_spec, points)

    campaign = args.output_root.resolve() / args.campaign_name
    campaign.mkdir(parents=True, exist_ok=True)
    config_path = campaign / "campaign_config.json"
    status_path = campaign / "campaign_status.json"
    if config_path.exists():
        if load_json(config_path) != config:
            raise ValueError("campaign configuration differs from the immutable saved config")
    else:
        atomic_write_json(config_path, config)

    jobs = []
    targets: set[Path] = set()
    for L in config["L_values"]:
        for point in points:
            h = float(point["h"])
            gamma_values = np.asarray(point["Gamma_values"], dtype=float)
            directory = campaign / "scans" / scan_directory_name(config, int(L), h)
            resolved = directory.resolve()
            validate_path_length(directory)
            if resolved in targets:
                raise ValueError("distinct jobs collide after formatting h")
            targets.add(resolved)

            complete, message = False, "missing"
            try:
                scan = validate_clean_ising_gamma_scan(
                    directory,
                    expected=expected(config, int(L), h),
                    require_mps=config["mps_save_policy"] == "selected",
                )
                if not np.allclose(
                    scan["arrays"]["Gamma_values"], gamma_values, rtol=0.0, atol=1e-14
                ):
                    raise ValueError("saved Gamma grid differs from targeted campaign config")
                complete, message = True, "compatible exact targeted grid"
            except (FileNotFoundError, ValueError, OSError, EOFError) as exc:
                if directory.exists():
                    message = f"incomplete or incompatible: {exc}"

            overwrite = has_any_artifact(directory) and not complete
            if overwrite and not args.rerun_incomplete:
                raise RuntimeError(
                    f"target {directory} is not reusable ({message}); "
                    "pass --rerun-incomplete"
                )
            job = {
                "L": int(L),
                "h": h,
                "Gamma_center": point["Gamma_center"],
                "Gamma_values": point["Gamma_values"],
                "Gamma_point_count": int(gamma_values.size),
                "status": "complete" if complete else "planned",
                "message": message,
                "target_directory": str(resolved),
            }
            if complete:
                job["source_directory"] = str(resolved)
            else:
                job["command"] = worker_command(
                    config, int(L), point, campaign, overwrite
                )
            jobs.append(job)

    status = {
        "schema_version": 2,
        "campaign_kind": "targeted_boundary",
        "campaign_name": args.campaign_name,
        "created_or_resumed_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "jobs": jobs,
    }
    atomic_write_json(status_path, status)
    complete_count = sum(job["status"] == "complete" for job in jobs)
    planned_count = sum(job["status"] == "planned" for job in jobs)
    planned_points = sum(
        job["Gamma_point_count"] for job in jobs if job["status"] == "planned"
    )
    print(
        f"Targeted campaign plan: complete={complete_count}, planned={planned_count}, "
        f"planned DMRG points={planned_points}"
    )
    if args.dry_run:
        for job in jobs:
            gamma_text = " ".join(f"{value:g}" for value in job["Gamma_values"])
            print(
                f"[{job['status']}] L={job['L']}, h={job['h']:g}, "
                f"Gamma=[{gamma_text}]: {job['message']}"
            )
            if "command" in job:
                print(subprocess.list2cmdline(job["command"]))
        return

    failures = 0
    for index, job in enumerate(jobs, start=1):
        if job["status"] == "complete":
            print(f"[{index}/{len(jobs)}] SKIP L={job['L']}, h={job['h']:g}")
            continue
        print(
            f"[{index}/{len(jobs)}] RUN L={job['L']}, h={job['h']:g}, "
            f"points={job['Gamma_point_count']}",
            flush=True,
        )
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
                scan = validate_clean_ising_gamma_scan(
                    directory,
                    expected=expected(config, int(job["L"]), float(job["h"])),
                    require_mps=config["mps_save_policy"] == "selected",
                )
                expected_gamma = np.asarray(job["Gamma_values"], dtype=float)
                if not np.allclose(
                    scan["arrays"]["Gamma_values"], expected_gamma, rtol=0.0, atol=1e-14
                ):
                    raise ValueError("saved Gamma grid differs from targeted campaign config")
                job["status"] = "complete"
                job["message"] = "compatible exact targeted grid"
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
    print(f"Targeted campaign complete: {campaign}")


if __name__ == "__main__":
    main()
