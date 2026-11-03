"""Self-contained Rényi-2 pair-information matrices for one clean Ising size."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .save_results import replace_with_retry


FORMAT = "clean_tfim_tn.clean_mixed_field_ising_R2I"
SCHEMA_VERSION = 1


def save_clean_ising_renyi2_information(
    path: str | Path,
    *,
    information_matrices: Any,
    single_site_entropies: Any,
    h_values: Any,
    Gamma_values: Any,
    branch_disagreement: Any,
    source_directories: Any,
    mps_sha256: Any,
    mps_manifest_sha256: Any,
    metadata: Mapping[str, Any],
) -> None:
    """Save R2I matrices with the same physical axes as clean topology C."""

    import h5py

    information = np.asarray(information_matrices, dtype=float)
    one_site = np.asarray(single_site_entropies, dtype=float)
    h = np.asarray(h_values, dtype=float)
    gamma = np.asarray(Gamma_values, dtype=float)
    disagreement = np.asarray(branch_disagreement, dtype=bool)
    sources = np.asarray(source_directories, dtype=str)
    mps_hashes = np.asarray(mps_sha256, dtype=str)
    manifest_hashes = np.asarray(mps_manifest_sha256, dtype=str)
    if information.ndim != 4 or information.shape[:2] != (h.size, gamma.size):
        raise ValueError("R2I must have shape (n_h,n_Gamma,L,L)")
    L = information.shape[2]
    if information.shape[3] != L:
        raise ValueError("R2I matrices must be square")
    if one_site.shape != (h.size, gamma.size, L):
        raise ValueError("single-site S2 must have shape (n_h,n_Gamma,L)")
    if disagreement.shape != (h.size, gamma.size):
        raise ValueError("invalid branch_disagreement shape")
    if (
        sources.shape != (h.size,)
        or mps_hashes.shape != (h.size,)
        or manifest_hashes.shape != (h.size,)
    ):
        raise ValueError("invalid provenance shape")
    for name, axis in (("h", h), ("Gamma", gamma)):
        if (
            axis.size < 1
            or np.any(~np.isfinite(axis))
            or np.any(np.diff(axis) <= 0.0)
        ):
            raise ValueError(f"{name} must be finite, sorted and distinct")
    if np.any(~np.isfinite(information)) or np.any(~np.isfinite(one_site)):
        raise ValueError("Rényi-2 data contain non-finite values")
    if not np.allclose(
        information,
        np.swapaxes(information, 2, 3),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("R2I must be symmetric")
    if not np.allclose(
        np.diagonal(information, axis1=2, axis2=3),
        0.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("R2I must have a zero diagonal")
    if np.any(information > 2.0 * np.log(2.0) + 1e-9):
        raise ValueError("R2I exceeds the two-qubit maximum 2 ln(2)")
    if np.any(one_site < -1e-10) or np.any(one_site > np.log(2.0) + 1e-9):
        raise ValueError("single-site S2 lies outside [0,ln(2)]")

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
                "R2I[i,j] = S2(rho_i) + S2(rho_j) - S2(rho_ij); "
                "R2I[i,i] = 0; natural logarithm"
            )
            h5file.attrs["L"] = L
            axes = h5file.create_group("axes")
            axes.create_dataset("h", data=h)
            axes.create_dataset("Gamma", data=gamma)
            data = h5file.create_group("data")
            data.create_dataset(
                "R2I",
                data=information,
                chunks=(1, 1, L, L),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
            )
            data.create_dataset(
                "single_site_S2",
                data=one_site,
                chunks=(1, 1, L),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
            )
            qc = h5file.create_group("qc")
            qc.create_dataset("branch_disagreement", data=disagreement)
            qc.attrs["R2I_minimum"] = float(np.min(information))
            qc.attrs["R2I_maximum"] = float(np.max(information))
            provenance = h5file.create_group("provenance")
            provenance.create_dataset(
                "source_directories",
                data=sources.astype(object),
                dtype=string_type,
            )
            provenance.create_dataset(
                "selected_mps_sha256",
                data=mps_hashes.astype(object),
                dtype=string_type,
            )
            provenance.create_dataset(
                "selected_mps_manifest_sha256",
                data=manifest_hashes.astype(object),
                dtype=string_type,
            )
            meta = h5file.create_group("metadata")
            meta.create_dataset(
                "json",
                data=json.dumps(dict(metadata), sort_keys=True),
                dtype=string_type,
            )
            h5file.flush()
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_clean_ising_renyi2_information(
    path: str | Path,
) -> dict[str, Any]:
    """Inspect the schema and dimensions without loading all matrices."""

    import h5py

    with h5py.File(Path(path), "r") as h5file:
        if h5file.attrs.get("format") != FORMAT:
            raise ValueError("not a clean mixed-field Ising R2I dataset")
        if int(h5file.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported clean R2I schema")
        shape = tuple(h5file["data/R2I"].shape)
        if tuple(h5file["data/single_site_S2"].shape) != (
            shape[0],
            shape[1],
            shape[2],
        ):
            raise ValueError("inconsistent single-site S2 shape")
        return {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "L": int(h5file.attrs["L"]),
            "shape": list(shape),
            "sample_count": int(shape[0] * shape[1]),
            "R2I_minimum": float(h5file["qc"].attrs["R2I_minimum"]),
            "R2I_maximum": float(h5file["qc"].attrs["R2I_maximum"]),
        }


def load_clean_ising_renyi2_information(
    path: str | Path,
) -> dict[str, Any]:
    """Load a validated clean R2I HDF5 dataset."""

    import h5py

    inspect_clean_ising_renyi2_information(path)
    with h5py.File(Path(path), "r") as h5file:
        def strings(name: str) -> np.ndarray:
            return np.asarray(h5file[name].asstr())

        return {
            "R2I": np.asarray(h5file["data/R2I"]),
            "single_site_S2": np.asarray(h5file["data/single_site_S2"]),
            "h_values": np.asarray(h5file["axes/h"]),
            "Gamma_values": np.asarray(h5file["axes/Gamma"]),
            "branch_disagreement": np.asarray(
                h5file["qc/branch_disagreement"]
            ),
            "source_directories": strings(
                "provenance/source_directories"
            ),
            "selected_mps_sha256": strings(
                "provenance/selected_mps_sha256"
            ),
            "selected_mps_manifest_sha256": strings(
                "provenance/selected_mps_manifest_sha256"
            ),
            "metadata": json.loads(h5file["metadata/json"].asstr()[()]),
        }
