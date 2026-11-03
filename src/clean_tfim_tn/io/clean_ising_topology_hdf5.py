"""Self-contained topology matrices for one clean mixed-field Ising size."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .save_results import replace_with_retry


FORMAT = "clean_tfim_tn.clean_mixed_field_ising_topology_D"
SCHEMA_VERSION = 1


def save_clean_ising_topology(
    path: str | Path,
    *,
    distance_matrices: Any,
    h_values: Any,
    Gamma_values: Any,
    branch_disagreement: Any,
    source_directories: Any,
    summary_sha256: Any,
    metadata: Mapping[str, Any],
) -> None:
    import h5py

    distance = np.asarray(distance_matrices, dtype=float)
    h = np.asarray(h_values, dtype=float)
    gamma = np.asarray(Gamma_values, dtype=float)
    disagreement = np.asarray(branch_disagreement, dtype=bool)
    sources = np.asarray(source_directories, dtype=str)
    hashes = np.asarray(summary_sha256, dtype=str)
    if distance.ndim != 4 or distance.shape[:2] != (h.size, gamma.size):
        raise ValueError("D must have shape (n_h,n_Gamma,L,L)")
    L = distance.shape[2]
    if distance.shape[3] != L:
        raise ValueError("D matrices must be square")
    if disagreement.shape != (h.size, gamma.size):
        raise ValueError("invalid branch_disagreement shape")
    if sources.shape != (h.size,) or hashes.shape != (h.size,):
        raise ValueError("invalid provenance shape")
    for name, axis in (("h", h), ("Gamma", gamma)):
        if axis.size < 1 or np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"{name} must be finite, sorted and distinct")
    if np.any(~np.isfinite(distance)):
        raise ValueError("D contains non-finite values")
    if not np.allclose(distance, np.swapaxes(distance, 2, 3), atol=1e-9, rtol=0.0):
        raise ValueError("D must be symmetric")
    if not np.allclose(np.diagonal(distance, axis1=2, axis2=3), 0.0, atol=1e-8, rtol=0.0):
        raise ValueError("D must have zero diagonal")
    if np.any(distance < -1e-9) or np.any(distance > 2.0 + 1e-9):
        raise ValueError("D must lie in [0,2]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    string_type = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(temporary, "w") as h5file:
            h5file.attrs["format"] = FORMAT
            h5file.attrs["schema_version"] = SCHEMA_VERSION
            h5file.attrs["axis_order"] = "h,Gamma,site_i,site_j"
            h5file.attrs["matrix_convention"] = "D[i,j] = 1 - Cstag[i,j]"
            h5file.attrs["L"] = L
            axes = h5file.create_group("axes")
            axes.create_dataset("h", data=h)
            axes.create_dataset("Gamma", data=gamma)
            data = h5file.create_group("data")
            data.create_dataset(
                "D", data=distance, chunks=(1, 1, L, L), compression="gzip",
                compression_opts=4, shuffle=True, fletcher32=True,
            )
            qc = h5file.create_group("qc")
            qc.create_dataset("branch_disagreement", data=disagreement)
            provenance = h5file.create_group("provenance")
            provenance.create_dataset("source_directories", data=sources.astype(object), dtype=string_type)
            provenance.create_dataset("summary_npz_sha256", data=hashes.astype(object), dtype=string_type)
            meta = h5file.create_group("metadata")
            meta.create_dataset("json", data=json.dumps(dict(metadata), sort_keys=True), dtype=string_type)
            h5file.flush()
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_clean_ising_topology(path: str | Path) -> dict[str, Any]:
    import h5py

    with h5py.File(Path(path), "r") as h5file:
        if h5file.attrs.get("format") != FORMAT:
            raise ValueError("not a clean mixed-field Ising topology dataset")
        if int(h5file.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported clean topology schema")
        shape = tuple(h5file["data/D"].shape)
        return {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "L": int(h5file.attrs["L"]),
            "shape": list(shape),
            "sample_count": int(shape[0] * shape[1]),
        }


def load_clean_ising_topology(path: str | Path) -> dict[str, Any]:
    import h5py

    inspect_clean_ising_topology(path)
    with h5py.File(Path(path), "r") as h5file:
        def strings(name: str) -> np.ndarray:
            return np.asarray(h5file[name].asstr())

        raw_meta = h5file["metadata/json"].asstr()[()]
        return {
            "D": np.asarray(h5file["data/D"]),
            "h_values": np.asarray(h5file["axes/h"]),
            "Gamma_values": np.asarray(h5file["axes/Gamma"]),
            "branch_disagreement": np.asarray(h5file["qc/branch_disagreement"]),
            "source_directories": strings("provenance/source_directories"),
            "summary_npz_sha256": strings("provenance/summary_npz_sha256"),
            "metadata": json.loads(raw_meta),
        }
