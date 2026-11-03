"""Persistence and validation for uniform mixed-field Ising Gamma scans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .clean_ising_mps_hdf5 import inspect_clean_ising_mps_collection
from .save_results import replace_with_retry, write_json


SUMMARY_NPZ = "continuation_summary.npz"
SUMMARY_JSON = "continuation_summary.json"
MPS_COLLECTION_FILE = "selected_mps.h5"
MPS_MANIFEST_FILE = "selected_mps_manifest.json"

REQUIRED_ARRAYS = {
    "Gamma_values", "h", "J", "J_values", "Gamma_by_point",
    "forward_energy", "backward_energy", "forward_energy_density",
    "backward_energy_density", "absolute_energy_difference",
    "branch_fidelity", "branch_disagreement", "forward_max_bond_dimension",
    "backward_max_bond_dimension", "selected_branch", "selected_energy",
    "selected_energy_density", "selected_max_bond_dimension",
    "half_chain_entropy", "entanglement_entropy", "bond_dimensions",
    "Czz", "Cstag", "D",
}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        write_json(temporary, payload)
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_clean_ising_gamma_scan(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    with np.load(root / SUMMARY_NPZ, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    payload = load_json(root / SUMMARY_JSON)
    if "metadata" not in payload:
        raise ValueError(f"{root} summary JSON has no metadata")
    return {
        "directory": root,
        "arrays": arrays,
        "payload": payload,
        "metadata": payload["metadata"],
    }


def _same_float(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-14))
    except (TypeError, ValueError):
        return False


def validate_clean_ising_gamma_scan(
    directory: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
    require_mps: bool | None = None,
    allow_qc_failures: bool = True,
) -> dict[str, Any]:
    scan = load_clean_ising_gamma_scan(directory)
    root, arrays, metadata = scan["directory"], scan["arrays"], scan["metadata"]
    if metadata.get("model") != "clean_mixed_field_ising":
        raise ValueError(f"{root} is not a clean mixed-field Ising scan")
    if metadata.get("scan_parameter") != "Gamma" or metadata.get("disorder") != "none":
        raise ValueError(f"{root} has invalid clean-model metadata")
    if expected:
        float_keys = {
            "J", "h", "max_E_err", "max_S_err", "svd_min",
            "branch_energy_tolerance", "branch_infidelity_tolerance",
            "selection_energy_tolerance",
        }
        for key, value in expected.items():
            actual = metadata.get(key)
            matches = _same_float(actual, value) if key in float_keys else actual == value
            if not matches:
                raise ValueError(f"{root} has inconsistent metadata {key}")
    missing = sorted(REQUIRED_ARRAYS.difference(arrays))
    if missing:
        raise ValueError(f"{root} is missing arrays: {', '.join(missing)}")
    gamma = np.asarray(arrays["Gamma_values"], dtype=float)
    L, n = int(metadata["L"]), gamma.size
    if n < 1 or np.any(~np.isfinite(gamma)) or np.any(np.diff(gamma) <= 0.0):
        raise ValueError(f"{root} has an invalid Gamma grid")
    if not np.array_equal(gamma, np.asarray(metadata.get("Gamma_values"), dtype=float)):
        raise ValueError(f"{root} has inconsistent JSON/NPZ Gamma grids")
    one_d = REQUIRED_ARRAYS.difference({
        "Gamma_values", "h", "J", "J_values", "Gamma_by_point",
        "entanglement_entropy", "bond_dimensions", "Czz", "Cstag", "D",
    })
    for key in one_d:
        if np.asarray(arrays[key]).shape != (n,):
            raise ValueError(f"{root} has invalid {key} shape")
    n_bonds = L - 1 if metadata.get("bc") in {"open", "obc"} else L
    if np.asarray(arrays["J_values"]).shape != (n_bonds,):
        raise ValueError(f"{root} has invalid J_values shape")
    if np.asarray(arrays["Gamma_by_point"]).shape != (n, L):
        raise ValueError(f"{root} has invalid Gamma_by_point shape")
    if np.asarray(arrays["entanglement_entropy"]).shape != (n, L - 1):
        raise ValueError(f"{root} has invalid entanglement_entropy shape")
    if np.asarray(arrays["bond_dimensions"]).shape != (n, L - 1):
        raise ValueError(f"{root} has invalid bond_dimensions shape")
    for key in ("Czz", "Cstag", "D"):
        if np.asarray(arrays[key]).shape != (n, L, L):
            raise ValueError(f"{root} has invalid {key} shape")
    numeric = REQUIRED_ARRAYS.difference({"selected_branch"})
    if any(not np.all(np.isfinite(np.asarray(arrays[key], dtype=float))) for key in numeric):
        raise ValueError(f"{root} contains non-finite selected data")
    if not _same_float(arrays["J"], metadata["J"]) or not _same_float(arrays["h"], metadata["h"]):
        raise ValueError(f"{root} has inconsistent scalar parameters")
    if not np.allclose(arrays["J_values"], metadata["J"], rtol=0.0, atol=1e-14):
        raise ValueError(f"{root} is not uniform in J")
    expected_gamma = np.broadcast_to(gamma[:, None], (n, L))
    if not np.allclose(arrays["Gamma_by_point"], expected_gamma, rtol=0.0, atol=1e-14):
        raise ValueError(f"{root} is not uniform in Gamma")
    selected_branch = np.asarray(arrays["selected_branch"]).astype(str)
    if np.any(~np.isin(selected_branch, ("forward", "backward"))):
        raise ValueError(f"{root} contains an unknown selected branch")
    fidelity = np.asarray(arrays["branch_fidelity"], dtype=float)
    if np.any((fidelity < -1e-12) | (fidelity > 1.0 + 1e-12)):
        raise ValueError(f"{root} contains invalid branch fidelities")
    czz = np.asarray(arrays["Czz"], dtype=float)
    cstag = np.asarray(arrays["Cstag"], dtype=float)
    distance = np.asarray(arrays["D"], dtype=float)
    parity = (-1.0) ** np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    if not np.allclose(czz, np.swapaxes(czz, 1, 2), atol=1e-9, rtol=0.0):
        raise ValueError(f"{root} has non-symmetric Czz")
    if not np.allclose(np.diagonal(czz, axis1=1, axis2=2), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{root} has invalid Czz diagonal")
    if not np.allclose(cstag, czz * parity, atol=1e-9, rtol=0.0):
        raise ValueError(f"{root} violates the Cstag convention")
    if not np.allclose(distance, 1.0 - cstag, atol=1e-9, rtol=0.0):
        raise ValueError(f"{root} violates D=1-Cstag")
    disagreement = np.asarray(arrays["branch_disagreement"], dtype=bool)
    if np.any(disagreement) and not allow_qc_failures:
        raise ValueError(f"{root} contains forward/backward disagreements")
    policy = metadata.get("mps_save_policy", "none")
    should_require = policy == "selected" if require_mps is None else require_mps
    if should_require:
        collection_path = root / MPS_COLLECTION_FILE
        manifest_path = root / MPS_MANIFEST_FILE
        if not collection_path.exists() or not manifest_path.exists():
            raise ValueError(f"{root} is missing selected-MPS artifacts")
        info = inspect_clean_ising_mps_collection(collection_path)
        manifest = load_json(manifest_path)
        if info["selected_state_count"] != n or len(manifest.get("point_records", [])) != n:
            raise ValueError(f"{root} has incomplete selected-MPS artifacts")
        for key in ("model", "L", "J", "h", "bc", "chi_max", "disorder"):
            left, right = manifest.get(key), metadata.get(key)
            matches = _same_float(left, right) if key in {"J", "h"} else left == right
            if not matches:
                raise ValueError(f"{root} has inconsistent MPS manifest {key}")
        if not np.array_equal(np.asarray(manifest.get("Gamma_values"), dtype=float), gamma):
            raise ValueError(f"{root} has inconsistent MPS Gamma grid")
    return scan
