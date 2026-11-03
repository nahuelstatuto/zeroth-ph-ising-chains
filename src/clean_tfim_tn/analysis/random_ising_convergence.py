"""Bond-dimension convergence for persisted random-Ising scans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


REQUIRED_SCAN_ARRAYS = (
    "h_values",
    "J",
    "Gamma",
    "forward_energy",
    "backward_energy",
    "half_chain_entropy",
    "Czz",
    "D",
    "branch_disagreement",
)


def _as_array(scan: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in scan:
        raise ValueError(f"scan is missing required array {key!r}")
    return np.asarray(scan[key])


def compare_random_ising_chi_scans(
    scans: Sequence[Mapping[str, np.ndarray]],
    chi_values: Sequence[int],
) -> tuple[dict, ...]:
    """Compare adjacent ``chi`` scans after validating their physical grid."""

    if len(scans) != len(chi_values) or len(scans) < 2:
        raise ValueError("provide the same number of scans and chi values, with at least two")
    pairs = sorted(zip(chi_values, scans), key=lambda item: item[0])
    chis = [int(chi) for chi, _scan in pairs]
    if any(chi < 1 for chi in chis) or len(set(chis)) != len(chis):
        raise ValueError("chi values must be positive and unique")

    reference = pairs[0][1]
    for key in REQUIRED_SCAN_ARRAYS:
        _as_array(reference, key)
    reference_h = _as_array(reference, "h_values").astype(float)
    reference_J = _as_array(reference, "J").astype(float)
    reference_Gamma = _as_array(reference, "Gamma").astype(float)
    L = reference_Gamma.size
    if reference_J.shape != (L - 1,):
        raise ValueError("J and Gamma lengths are inconsistent with an open chain")
    expected_matrix_shape = (reference_h.size, L, L)

    for _chi, scan in pairs:
        for key in REQUIRED_SCAN_ARRAYS:
            _as_array(scan, key)
        for key, expected in (
            ("h_values", reference_h),
            ("J", reference_J),
            ("Gamma", reference_Gamma),
        ):
            actual = _as_array(scan, key)
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError(f"scans have inconsistent {key}")
        if _as_array(scan, "Czz").shape != expected_matrix_shape:
            raise ValueError("Czz has an inconsistent shape")
        if _as_array(scan, "D").shape != expected_matrix_shape:
            raise ValueError("D has an inconsistent shape")
        if _as_array(scan, "half_chain_entropy").shape != reference_h.shape:
            raise ValueError("half_chain_entropy has an inconsistent shape")
        for key in ("forward_energy", "backward_energy", "branch_disagreement"):
            if _as_array(scan, key).shape != reference_h.shape:
                raise ValueError(f"{key} has an inconsistent shape")

    comparisons = []
    for (chi_lower, lower), (chi_upper, upper) in zip(pairs, pairs[1:]):
        lower_energy = np.minimum(
            _as_array(lower, "forward_energy"), _as_array(lower, "backward_energy")
        ).astype(float)
        upper_energy = np.minimum(
            _as_array(upper, "forward_energy"), _as_array(upper, "backward_energy")
        ).astype(float)
        energy_density_change = np.abs(upper_energy - lower_energy) / L
        entropy_change = np.abs(
            _as_array(upper, "half_chain_entropy").astype(float)
            - _as_array(lower, "half_chain_entropy").astype(float)
        )
        czz_change = _as_array(upper, "Czz").astype(float) - _as_array(
            lower, "Czz"
        ).astype(float)
        distance_change = _as_array(upper, "D").astype(float) - _as_array(
            lower, "D"
        ).astype(float)
        axes = (1, 2)
        max_czz_change = np.max(np.abs(czz_change), axis=axes)
        rms_czz_change = np.sqrt(np.mean(czz_change**2, axis=axes))
        max_distance_change = np.max(np.abs(distance_change), axis=axes)
        rms_distance_change = np.sqrt(np.mean(distance_change**2, axis=axes))
        either_branch_disagrees = np.logical_or(
            _as_array(lower, "branch_disagreement").astype(bool),
            _as_array(upper, "branch_disagreement").astype(bool),
        )
        comparisons.append(
            {
                "chi_lower": int(chi_lower),
                "chi_upper": int(chi_upper),
                "h_values": reference_h,
                "absolute_energy_density_change": energy_density_change,
                "absolute_half_chain_entropy_change": entropy_change,
                "max_abs_Czz_change": max_czz_change,
                "rms_Czz_change": rms_czz_change,
                "max_abs_D_change": max_distance_change,
                "rms_D_change": rms_distance_change,
                "either_branch_disagrees": either_branch_disagrees,
                "maxima": {
                    "absolute_energy_density_change": float(np.max(energy_density_change)),
                    "absolute_half_chain_entropy_change": float(np.max(entropy_change)),
                    "max_abs_Czz_change": float(np.max(max_czz_change)),
                    "rms_Czz_change": float(np.max(rms_czz_change)),
                    "max_abs_D_change": float(np.max(max_distance_change)),
                    "rms_D_change": float(np.max(rms_distance_change)),
                },
            }
        )
    return tuple(comparisons)
