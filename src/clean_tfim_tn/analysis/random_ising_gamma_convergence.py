"""Bond-dimension convergence for random-Ising scans over Gamma0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


REQUIRED_ARRAYS = (
    "Gamma0_values", "u", "v", "J", "Gamma", "forward_energy",
    "backward_energy", "half_chain_entropy", "Czz", "Cstag", "D",
    "branch_disagreement",
)


def _array(scan: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in scan:
        raise ValueError(f"scan is missing required array {key!r}")
    return np.asarray(scan[key])


def compare_random_ising_gamma_chi_scans(
    scans: Sequence[Mapping[str, np.ndarray]],
    chi_values: Sequence[int],
) -> tuple[dict, ...]:
    """Compare adjacent-chi Gamma0 scans after validating their physical grid."""

    if len(scans) != len(chi_values) or len(scans) < 2:
        raise ValueError("provide the same number of scans and chi values, with at least two")
    pairs = sorted(zip(chi_values, scans), key=lambda item: item[0])
    chis = [int(chi) for chi, _scan in pairs]
    if any(chi < 1 for chi in chis) or len(set(chis)) != len(chis):
        raise ValueError("chi values must be positive and unique")

    reference = pairs[0][1]
    for key in REQUIRED_ARRAYS:
        _array(reference, key)
    gamma0 = _array(reference, "Gamma0_values").astype(float)
    reference_u = _array(reference, "u").astype(float)
    reference_v = _array(reference, "v").astype(float)
    reference_J = _array(reference, "J").astype(float)
    reference_Gamma = _array(reference, "Gamma").astype(float)
    L = reference_v.size
    if reference_J.shape != (L - 1,) or reference_u.shape != (L - 1,):
        raise ValueError("u, v and J lengths are inconsistent with an open chain")
    if reference_Gamma.shape != (gamma0.size, L):
        raise ValueError("Gamma has an inconsistent shape")
    matrix_shape = (gamma0.size, L, L)

    for _chi, scan in pairs:
        for key in REQUIRED_ARRAYS:
            _array(scan, key)
        for key, expected in (
            ("Gamma0_values", gamma0), ("u", reference_u), ("v", reference_v),
            ("J", reference_J), ("Gamma", reference_Gamma),
        ):
            actual = _array(scan, key)
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError(f"scans have inconsistent {key}")
        for key in ("Czz", "Cstag", "D"):
            if _array(scan, key).shape != matrix_shape:
                raise ValueError(f"{key} has an inconsistent shape")
        for key in (
            "forward_energy", "backward_energy", "half_chain_entropy",
            "branch_disagreement",
        ):
            if _array(scan, key).shape != gamma0.shape:
                raise ValueError(f"{key} has an inconsistent shape")

    comparisons = []
    axes = (1, 2)
    for (chi_lower, lower), (chi_upper, upper) in zip(pairs, pairs[1:]):
        lower_energy = np.minimum(
            _array(lower, "forward_energy"), _array(lower, "backward_energy")
        ).astype(float)
        upper_energy = np.minimum(
            _array(upper, "forward_energy"), _array(upper, "backward_energy")
        ).astype(float)
        energy_change = np.abs(upper_energy - lower_energy) / L
        entropy_change = np.abs(
            _array(upper, "half_chain_entropy").astype(float)
            - _array(lower, "half_chain_entropy").astype(float)
        )
        czz_change = _array(upper, "Czz").astype(float) - _array(lower, "Czz").astype(float)
        distance_change = _array(upper, "D").astype(float) - _array(lower, "D").astype(float)
        lower_ms2 = np.sum(_array(lower, "Cstag"), axis=axes) / L**2
        upper_ms2 = np.sum(_array(upper, "Cstag"), axis=axes) / L**2
        lower_m02 = np.sum(_array(lower, "Czz"), axis=axes) / L**2
        upper_m02 = np.sum(_array(upper, "Czz"), axis=axes) / L**2
        values = {
            "absolute_energy_density_change": energy_change,
            "absolute_half_chain_entropy_change": entropy_change,
            "max_abs_Czz_change": np.max(np.abs(czz_change), axis=axes),
            "rms_Czz_change": np.sqrt(np.mean(czz_change**2, axis=axes)),
            "max_abs_D_change": np.max(np.abs(distance_change), axis=axes),
            "rms_D_change": np.sqrt(np.mean(distance_change**2, axis=axes)),
            "absolute_ms2_change": np.abs(upper_ms2 - lower_ms2),
            "absolute_m02_change": np.abs(upper_m02 - lower_m02),
        }
        comparisons.append({
            "chi_lower": int(chi_lower),
            "chi_upper": int(chi_upper),
            "Gamma0_values": gamma0,
            **values,
            "either_branch_disagrees": np.logical_or(
                _array(lower, "branch_disagreement").astype(bool),
                _array(upper, "branch_disagreement").astype(bool),
            ),
            "maxima": {key: float(np.max(value)) for key, value in values.items()},
        })
    return tuple(comparisons)
