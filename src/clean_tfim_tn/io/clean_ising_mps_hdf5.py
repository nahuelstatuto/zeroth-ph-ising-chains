"""Atomic MPS collections for uniform mixed-field Ising scans."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .save_results import replace_with_retry


FORMAT = "clean_tfim_tn.clean_mixed_field_ising_mps_collection"
SCHEMA_VERSION = 1


def _states(name: str, values: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(values or {})
    if any(not isinstance(key, str) or not key for key in result):
        raise ValueError(f"{name} keys must be non-empty strings")
    if any(value is None for value in result.values()):
        raise ValueError(f"{name} must not contain None states")
    return result


def save_clean_ising_mps_collection(
    path: str | Path,
    selected_states: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    disagreement_forward_states: Mapping[str, Any] | None = None,
    disagreement_backward_states: Mapping[str, Any] | None = None,
) -> None:
    import h5py
    from tenpy.tools import hdf5_io

    selected = _states("selected_states", selected_states)
    forward = _states("disagreement_forward_states", disagreement_forward_states)
    backward = _states("disagreement_backward_states", disagreement_backward_states)
    if not selected:
        raise ValueError("selected_states must not be empty")
    if set(forward) != set(backward) or not set(forward).issubset(selected):
        raise ValueError("inconsistent disagreement state keys")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(temporary, "w") as h5file:
            h5file.attrs["format"] = FORMAT
            h5file.attrs["schema_version"] = SCHEMA_VERSION
            h5file.attrs["selected_state_count"] = len(selected)
            h5file.attrs["disagreement_point_count"] = len(forward)
            hdf5_io.save_to_hdf5(h5file, {
                "selected_states": selected,
                "disagreement_forward_states": forward,
                "disagreement_backward_states": backward,
                "metadata": dict(metadata),
            }, path="/payload")
            h5file.flush()
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_clean_ising_mps_collection(path: str | Path) -> dict[str, Any]:
    import h5py

    with h5py.File(Path(path), "r") as h5file:
        if h5file.attrs.get("format") != FORMAT:
            raise ValueError("not a clean mixed-field Ising MPS collection")
        if int(h5file.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported clean MPS schema")
        if "payload" not in h5file:
            raise ValueError("incomplete clean MPS collection")
        selected = int(h5file.attrs.get("selected_state_count", -1))
        disagreement = int(h5file.attrs.get("disagreement_point_count", -1))
        if selected < 1 or disagreement < 0:
            raise ValueError("invalid clean MPS collection counts")
        return {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "selected_state_count": selected,
            "disagreement_point_count": disagreement,
        }


def load_clean_ising_mps_collection(path: str | Path) -> dict[str, Any]:
    import h5py
    from tenpy.tools import hdf5_io

    inspect_clean_ising_mps_collection(path)
    with h5py.File(Path(path), "r") as h5file:
        payload = hdf5_io.load_from_hdf5(h5file, path="/payload")
    required = {
        "selected_states", "disagreement_forward_states",
        "disagreement_backward_states", "metadata",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("incomplete clean MPS payload")
    selected = _states("selected_states", payload["selected_states"])
    forward = _states("disagreement_forward_states", payload["disagreement_forward_states"])
    backward = _states("disagreement_backward_states", payload["disagreement_backward_states"])
    if set(forward) != set(backward) or not set(forward).issubset(selected):
        raise ValueError("inconsistent clean MPS payload")
    return {
        "selected_states": selected,
        "disagreement_forward_states": forward,
        "disagreement_backward_states": backward,
        "metadata": dict(payload["metadata"]),
    }
