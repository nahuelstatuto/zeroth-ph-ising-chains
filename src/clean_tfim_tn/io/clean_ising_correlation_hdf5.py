"""Self-contained staggered-correlation matrices for one clean Ising size."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .save_results import replace_with_retry


FORMAT = "clean_tfim_tn.clean_mixed_field_ising_correlation_C"
SCHEMA_VERSION = 1


def save_clean_ising_correlation(
    path: str | Path,
    *,
    correlation_matrices: Any,
    h_values: Any,
    Gamma_values: Any,
    branch_disagreement: Any,
    source_directories: Any,
    summary_sha256: Any,
    metadata: Mapping[str, Any],
) -> None:
    import h5py

    correlation = np.asarray(correlation_matrices, dtype=float)
    h = np.asarray(h_values, dtype=float)
    gamma = np.asarray(Gamma_values, dtype=float)
    disagreement = np.asarray(branch_disagreement, dtype=bool)
    sources = np.asarray(source_directories, dtype=str)
    hashes = np.asarray(summary_sha256, dtype=str)
    if correlation.ndim != 4 or correlation.shape[:2] != (h.size, gamma.size):
        raise ValueError("C must have shape (n_h,n_Gamma,L,L)")
    L = correlation.shape[2]
    if correlation.shape[3] != L:
        raise ValueError("C matrices must be square")
    if disagreement.shape != (h.size, gamma.size):
        raise ValueError("invalid branch_disagreement shape")
    if sources.shape != (h.size,) or hashes.shape != (h.size,):
        raise ValueError("invalid provenance shape")
    for name, axis in (("h", h), ("Gamma", gamma)):
        if axis.size < 1 or np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"{name} must be finite, sorted and distinct")
    if np.any(~np.isfinite(correlation)):
        raise ValueError("C contains non-finite values")
    if not np.allclose(
        correlation, np.swapaxes(correlation, 2, 3), atol=1e-9, rtol=0.0
    ):
        raise ValueError("C must be symmetric")
    if not np.allclose(
        np.diagonal(correlation, axis1=2, axis2=3), 1.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("C must have unit diagonal")
    if np.any(correlation < -1.0 - 1e-9) or np.any(correlation > 1.0 + 1e-9):
        raise ValueError("C must lie in [-1,1]")

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
            h5file.attrs["matrix_convention"] = (
                "C[i,j] = Cstag[i,j] = (-1)^abs(i-j) <sigma_z_i sigma_z_j>"
            )
            h5file.attrs["L"] = L
            axes = h5file.create_group("axes")
            axes.create_dataset("h", data=h)
            axes.create_dataset("Gamma", data=gamma)
            data = h5file.create_group("data")
            data.create_dataset(
                "C",
                data=correlation,
                chunks=(1, 1, L, L),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
            )
            qc = h5file.create_group("qc")
            qc.create_dataset("branch_disagreement", data=disagreement)
            provenance = h5file.create_group("provenance")
            provenance.create_dataset(
                "source_directories", data=sources.astype(object), dtype=string_type
            )
            provenance.create_dataset(
                "summary_npz_sha256", data=hashes.astype(object), dtype=string_type
            )
            meta = h5file.create_group("metadata")
            meta.create_dataset(
                "json", data=json.dumps(dict(metadata), sort_keys=True), dtype=string_type
            )
            h5file.flush()
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_clean_ising_correlation(path: str | Path) -> dict[str, Any]:
    import h5py

    with h5py.File(Path(path), "r") as h5file:
        if h5file.attrs.get("format") != FORMAT:
            raise ValueError("not a clean mixed-field Ising correlation dataset")
        if int(h5file.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported clean correlation schema")
        shape = tuple(h5file["data/C"].shape)
        return {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "L": int(h5file.attrs["L"]),
            "shape": list(shape),
            "sample_count": int(shape[0] * shape[1]),
        }


def load_clean_ising_correlation(path: str | Path) -> dict[str, Any]:
    import h5py

    inspect_clean_ising_correlation(path)
    with h5py.File(Path(path), "r") as h5file:
        def strings(name: str) -> np.ndarray:
            return np.asarray(h5file[name].asstr())

        raw_meta = h5file["metadata/json"].asstr()[()]
        return {
            "C": np.asarray(h5file["data/C"]),
            "h_values": np.asarray(h5file["axes/h"]),
            "Gamma_values": np.asarray(h5file["axes/Gamma"]),
            "branch_disagreement": np.asarray(h5file["qc/branch_disagreement"]),
            "source_directories": strings("provenance/source_directories"),
            "summary_sha256": strings("provenance/summary_npz_sha256"),
            "metadata": json.loads(raw_meta),
        }
