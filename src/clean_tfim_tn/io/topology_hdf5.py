"""Self-contained HDF5 datasets of distance matrices for topology analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .save_results import replace_with_retry


TOPOLOGY_D_FORMAT = "clean_tfim_tn.random_ising_topology_D"
TOPOLOGY_D_SCHEMA_VERSION = 1


def _validate_axes(
    h_values: np.ndarray, seeds: np.ndarray, gamma0_values: np.ndarray
) -> None:
    for name, values in (("h_values", h_values), ("Gamma0_values", gamma0_values)):
        if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError(f"{name} must be sorted and distinct")
    if seeds.ndim != 1 or seeds.size < 1 or np.any(np.diff(seeds) <= 0):
        raise ValueError("seeds must be a non-empty sorted distinct array")


def _validated_payload(
    distance_matrices: Any,
    h_values: Any,
    seeds: Any,
    gamma0_values: Any,
    branch_disagreement: Any,
    source_directories: Any,
    summary_sha256: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(h_values, dtype=float)
    seed = np.asarray(seeds, dtype=np.int64)
    gamma0 = np.asarray(gamma0_values, dtype=float)
    _validate_axes(h, seed, gamma0)

    distance = np.asarray(distance_matrices, dtype=float)
    if distance.ndim != 5 or distance.shape[-1] != distance.shape[-2]:
        raise ValueError("distance_matrices must have shape (h, seed, Gamma0, L, L)")
    L = distance.shape[-1]
    expected = (h.size, seed.size, gamma0.size, L, L)
    if distance.shape != expected:
        raise ValueError(f"distance_matrices has shape {distance.shape}; expected {expected}")
    if L < 2 or not np.all(np.isfinite(distance)):
        raise ValueError("distance_matrices must be finite matrices with L at least 2")
    if not np.allclose(distance, np.swapaxes(distance, -1, -2), rtol=0.0, atol=1e-9):
        raise ValueError("distance_matrices must be symmetric")
    if not np.allclose(
        np.diagonal(distance, axis1=-2, axis2=-1), 0.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError("distance_matrices must have zero diagonal")
    if np.any(distance < -1e-9) or np.any(distance > 2.0 + 1e-9):
        raise ValueError("distance_matrices must lie in [0, 2] within tolerance")

    disagreement = np.asarray(branch_disagreement, dtype=bool)
    if disagreement.shape != expected[:3]:
        raise ValueError("branch_disagreement must have shape (h, seed, Gamma0)")
    sources = np.asarray(source_directories, dtype=str)
    hashes = np.asarray(summary_sha256, dtype=str)
    if sources.shape != expected[:2] or hashes.shape != expected[:2]:
        raise ValueError("source directories and hashes must have shape (h, seed)")
    if np.any(sources == ""):
        raise ValueError("source directories must not be empty")
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
           for value in hashes.flat):
        raise ValueError("summary hashes must be lowercase SHA-256 hex digests")
    return distance, h, seed, gamma0, disagreement, sources, hashes


def save_topology_distance_dataset_hdf5(
    path: str | Path,
    *,
    distance_matrices: Any,
    h_values: Any,
    seeds: Any,
    gamma0_values: Any,
    branch_disagreement: Any,
    source_directories: Any,
    summary_sha256: Any,
    metadata: Mapping[str, Any],
) -> None:
    """Atomically save an indexed collection of topology distance matrices."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("Saving a topology dataset requires h5py.") from exc

    distance, h, seed, gamma0, disagreement, sources, hashes = _validated_payload(
        distance_matrices,
        h_values,
        seeds,
        gamma0_values,
        branch_disagreement,
        source_directories,
        summary_sha256,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(temporary, "w") as h5file:
            h5file.attrs["format"] = TOPOLOGY_D_FORMAT
            h5file.attrs["schema_version"] = TOPOLOGY_D_SCHEMA_VERSION
            h5file.attrs["axis_order"] = "h,seed,Gamma0,site_i,site_j"
            h5file.attrs["matrix_convention"] = "D[i,j] = 1 - Cstag[i,j]"
            h5file.attrs["L"] = distance.shape[-1]
            h5file.attrs["sample_count"] = int(np.prod(distance.shape[:3]))
            axes = h5file.create_group("axes")
            axes.create_dataset("h", data=h)
            axes.create_dataset("seed", data=seed)
            axes.create_dataset("Gamma0", data=gamma0)
            data = h5file.create_group("data")
            data.create_dataset(
                "D",
                data=distance,
                chunks=(1, 1, 1, distance.shape[-1], distance.shape[-1]),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
            )
            qc = h5file.create_group("qc")
            qc.create_dataset("branch_disagreement", data=disagreement)
            provenance = h5file.create_group("provenance")
            provenance.create_dataset(
                "source_directory", data=sources.astype(object), dtype=string_dtype
            )
            provenance.create_dataset(
                "summary_npz_sha256", data=hashes.astype(object), dtype=string_dtype
            )
            metadata_group = h5file.create_group("metadata")
            metadata_group.create_dataset(
                "json",
                data=json.dumps(dict(metadata), sort_keys=True),
                dtype=string_dtype,
            )
            h5file.flush()
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_topology_distance_dataset_hdf5(path: str | Path) -> dict[str, Any]:
    """Inspect schema and dimensions without loading the distance matrices."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("Inspecting a topology dataset requires h5py.") from exc
    with h5py.File(Path(path), "r") as h5file:
        if h5file.attrs.get("format") != TOPOLOGY_D_FORMAT:
            raise ValueError("HDF5 file is not a random-Ising topology D dataset")
        schema = int(h5file.attrs.get("schema_version", -1))
        if schema != TOPOLOGY_D_SCHEMA_VERSION:
            raise ValueError(f"unsupported topology D schema {schema}")
        required = ("axes/h", "axes/seed", "axes/Gamma0", "data/D")
        if any(name not in h5file for name in required):
            raise ValueError("topology D dataset is missing required arrays")
        shape = tuple(int(value) for value in h5file["data/D"].shape)
        if len(shape) != 5 or shape[-1] != shape[-2]:
            raise ValueError("topology D dataset has an invalid matrix shape")
        return {
            "format": TOPOLOGY_D_FORMAT,
            "schema_version": schema,
            "shape": shape,
            "L": int(h5file.attrs["L"]),
            "sample_count": int(h5file.attrs["sample_count"]),
            "h_count": int(h5file["axes/h"].shape[0]),
            "seed_count": int(h5file["axes/seed"].shape[0]),
            "Gamma0_count": int(h5file["axes/Gamma0"].shape[0]),
        }


def load_topology_distance_dataset_hdf5(path: str | Path) -> dict[str, Any]:
    """Load a trusted topology D dataset into NumPy arrays."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("Loading a topology dataset requires h5py.") from exc
    inspect_topology_distance_dataset_hdf5(path)
    with h5py.File(Path(path), "r") as h5file:
        return {
            "D": np.asarray(h5file["data/D"]),
            "h_values": np.asarray(h5file["axes/h"]),
            "seeds": np.asarray(h5file["axes/seed"]),
            "Gamma0_values": np.asarray(h5file["axes/Gamma0"]),
            "branch_disagreement": np.asarray(h5file["qc/branch_disagreement"], dtype=bool),
            "source_directories": np.asarray(h5file["provenance/source_directory"].asstr()),
            "summary_sha256": np.asarray(h5file["provenance/summary_npz_sha256"].asstr()),
            "metadata": json.loads(h5file["metadata/json"].asstr()[()]),
        }
