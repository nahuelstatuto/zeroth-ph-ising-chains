"""Native TeNPy MPS persistence using its HDF5 serialization format."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .save_results import replace_with_retry


MPS_COLLECTION_FORMAT = "clean_tfim_tn.random_ising_mps_collection"
MPS_COLLECTION_SCHEMA_VERSION = 1


def save_mps_hdf5(path: str | Path, state: Any, metadata: Mapping[str, Any] | None = None) -> None:
    """Save a finite TeNPy MPS and lightweight metadata to HDF5."""

    try:
        import h5py
        from tenpy.tools import hdf5_io
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("Saving a TeNPy MPS requires tenpy and h5py.") from exc

    payload = {"psi": state, "metadata": dict(metadata or {})}
    with h5py.File(Path(path), "w") as h5file:
        hdf5_io.save_to_hdf5(h5file, payload)


def load_mps_hdf5(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load an MPS saved by :func:`save_mps_hdf5`.

    Only load trusted files: TeNPy's loader reconstructs Python objects and
    therefore has security implications comparable to unpickling.
    """

    try:
        import h5py
        from tenpy.tools import hdf5_io
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("Loading a TeNPy MPS requires tenpy and h5py.") from exc

    with h5py.File(Path(path), "r") as h5file:
        payload = hdf5_io.load_from_hdf5(h5file)
    if not isinstance(payload, dict) or "psi" not in payload:
        raise ValueError("HDF5 file does not contain a saved TeNPy MPS")
    return payload["psi"], dict(payload.get("metadata", {}))


def _validate_state_mapping(name: str, states: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(states)
    if any(not isinstance(key, str) or not key for key in result):
        raise ValueError(f"{name} keys must be non-empty strings")
    if any(state is None for state in result.values()):
        raise ValueError(f"{name} must not contain None states")
    return result


def save_mps_collection_hdf5(
    path: str | Path,
    selected_states: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    disagreement_forward_states: Mapping[str, Any] | None = None,
    disagreement_backward_states: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save selected random-Ising MPS states for one scan.

    The main mapping contains exactly one accepted state per physical point.
    Optional branch mappings are intended only for points where the forward and
    backward continuations disagree. Keys are stable point identifiers stored
    separately in the lightweight JSON manifest.
    """

    try:
        import h5py
        from tenpy.tools import hdf5_io
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("Saving a TeNPy MPS collection requires tenpy and h5py.") from exc

    selected = _validate_state_mapping("selected_states", selected_states)
    if not selected:
        raise ValueError("selected_states must not be empty")
    forward = _validate_state_mapping(
        "disagreement_forward_states", disagreement_forward_states or {}
    )
    backward = _validate_state_mapping(
        "disagreement_backward_states", disagreement_backward_states or {}
    )
    if set(forward) != set(backward):
        raise ValueError("forward/backward disagreement state keys must match")
    if not set(forward).issubset(selected):
        raise ValueError("disagreement state keys must identify selected scan points")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    payload = {
        "selected_states": selected,
        "disagreement_forward_states": forward,
        "disagreement_backward_states": backward,
        "metadata": dict(metadata),
    }
    try:
        with h5py.File(temporary, "w") as h5file:
            h5file.attrs["format"] = MPS_COLLECTION_FORMAT
            h5file.attrs["schema_version"] = MPS_COLLECTION_SCHEMA_VERSION
            h5file.attrs["selected_state_count"] = len(selected)
            h5file.attrs["disagreement_point_count"] = len(forward)
            hdf5_io.save_to_hdf5(h5file, payload, path="/payload")
            h5file.flush()
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_mps_collection_hdf5(path: str | Path) -> dict[str, Any]:
    """Read lightweight collection attributes without reconstructing any MPS."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("Inspecting a TeNPy MPS collection requires h5py.") from exc

    with h5py.File(Path(path), "r") as h5file:
        if h5file.attrs.get("format") != MPS_COLLECTION_FORMAT:
            raise ValueError("HDF5 file is not a random-Ising MPS collection")
        schema_version = int(h5file.attrs.get("schema_version", -1))
        if schema_version != MPS_COLLECTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported MPS collection schema {schema_version}")
        if "payload" not in h5file:
            raise ValueError("MPS collection is missing its payload")
        selected_count = int(h5file.attrs.get("selected_state_count", -1))
        disagreement_count = int(h5file.attrs.get("disagreement_point_count", -1))
        if selected_count < 1 or disagreement_count < 0:
            raise ValueError("MPS collection has invalid state counts")
        return {
            "format": MPS_COLLECTION_FORMAT,
            "schema_version": schema_version,
            "selected_state_count": selected_count,
            "disagreement_point_count": disagreement_count,
        }


def load_mps_collection_hdf5(path: str | Path) -> dict[str, Any]:
    """Load a trusted random-Ising MPS collection saved by this module."""

    try:
        import h5py
        from tenpy.tools import hdf5_io
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError("Loading a TeNPy MPS collection requires tenpy and h5py.") from exc

    inspect_mps_collection_hdf5(path)
    with h5py.File(Path(path), "r") as h5file:
        payload = hdf5_io.load_from_hdf5(h5file, path="/payload")
    if not isinstance(payload, dict):
        raise ValueError("MPS collection payload is not a dictionary")
    required = {
        "selected_states",
        "disagreement_forward_states",
        "disagreement_backward_states",
        "metadata",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("MPS collection is missing: " + ", ".join(missing))
    selected = _validate_state_mapping("selected_states", payload["selected_states"])
    forward = _validate_state_mapping(
        "disagreement_forward_states", payload["disagreement_forward_states"]
    )
    backward = _validate_state_mapping(
        "disagreement_backward_states", payload["disagreement_backward_states"]
    )
    if set(forward) != set(backward) or not set(forward).issubset(selected):
        raise ValueError("MPS collection contains inconsistent disagreement keys")
    return {
        "selected_states": selected,
        "disagreement_forward_states": forward,
        "disagreement_backward_states": backward,
        "metadata": dict(payload["metadata"]),
    }
