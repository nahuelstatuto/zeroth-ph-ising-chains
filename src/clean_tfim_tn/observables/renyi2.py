"""Rényi-2 entropies and pair-information matrices from finite TeNPy MPS."""

from __future__ import annotations

from typing import Any

import numpy as np


PAULI_OPERATORS = ("Sigmax", "Sigmay", "Sigmaz")


def _bounded_purity(values: np.ndarray, *, tolerance: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError("Rényi-2 purity contains non-finite values")
    if np.any(array < -tolerance) or np.any(array > 1.0 + tolerance):
        raise ValueError("Rényi-2 purity lies outside [0,1]")
    return np.clip(array, np.finfo(float).tiny, 1.0)


def half_chain_renyi2_entropy(state: Any) -> float:
    """Return the central-cut Rényi-2 entropy of an even finite MPS."""

    L = int(getattr(state, "L", 0))
    if L < 2 or L % 2 != 0 or getattr(state, "bc", None) != "finite":
        raise ValueError(
            "half-chain Rényi-2 entropy requires an even finite MPS with L >= 2"
        )
    entropies = np.asarray(state.entanglement_entropy(n=2), dtype=float)
    if entropies.shape != (L - 1,) or np.any(~np.isfinite(entropies)):
        raise ValueError("MPS returned invalid Rényi-2 bond entropies")
    return float(entropies[(L // 2) - 1])


def renyi2_information_matrix(
    state: Any,
    *,
    numerical_tolerance: float = 1e-10,
    assume_real_wavefunction: bool = False,
) -> dict[str, np.ndarray | float]:
    r"""Compute the pairwise Rényi-2 information matrix of a qubit MPS.

    For distinct sites,

    .. math::

        I_2(i:j) = S_2(\rho_i) + S_2(\rho_j) - S_2(\rho_{ij}),
        \qquad S_2(\rho)=-\ln\operatorname{Tr}\rho^2.

    Two-qubit purities are reconstructed from the complete Pauli basis,

    .. math::

        \operatorname{Tr}\rho_{ij}^2 =
        \frac{1}{4}\left[
        1 + \sum_a\langle\sigma_a^i\rangle^2
          + \sum_b\langle\sigma_b^j\rangle^2
          + \sum_{ab}\langle\sigma_a^i\sigma_b^j\rangle^2
        \right].

    This avoids building one reduced density matrix per pair. The diagonal of
    ``R2I`` is defined as zero because it is a pair dissimilarity input
    convention, not ``I_2(i:i)``.
    """

    L = int(getattr(state, "L", 0))
    if L < 2 or getattr(state, "bc", None) != "finite":
        raise ValueError("Rényi-2 information requires a finite MPS with L >= 2")
    if any(int(getattr(site, "dim", 0)) != 2 for site in state.sites):
        raise ValueError("Rényi-2 Pauli reconstruction requires qubit sites")

    one_point = np.stack(
        [
            np.asarray(state.expectation_value(operator), dtype=np.complex128)
            for operator in PAULI_OPERATORS
        ]
    )
    if one_point.shape != (3, L):
        raise ValueError("MPS returned invalid one-site expectation values")
    if np.max(np.abs(one_point.imag)) > numerical_tolerance:
        raise ValueError("Hermitian one-site expectations have a large imaginary part")
    one_point = one_point.real

    correlations: dict[tuple[str, str], np.ndarray] = {}
    maximum_correlation_imaginary_part = 0.0
    if assume_real_wavefunction:
        if np.max(np.abs(one_point[1])) > numerical_tolerance:
            raise ValueError(
                "real-wavefunction optimization requires <Sigmay> = 0"
            )
        operator_pairs = (
            ("Sigmax", "Sigmax"),
            ("Sigmax", "Sigmaz"),
            ("Sigmay", "Sigmay"),
            ("Sigmaz", "Sigmaz"),
        )
    else:
        operator_pairs = tuple(
            (first, second)
            for first_index, first in enumerate(PAULI_OPERATORS)
            for second in PAULI_OPERATORS[first_index:]
        )
    for first, second in operator_pairs:
        values = np.asarray(
            state.correlation_function(
                first,
                second,
                hermitian=first == second,
            ),
            dtype=np.complex128,
        )
        if values.shape != (L, L):
            raise ValueError("MPS returned an invalid Pauli correlation matrix")
        off_diagonal = values[~np.eye(L, dtype=bool)]
        maximum_correlation_imaginary_part = max(
            maximum_correlation_imaginary_part,
            float(np.max(np.abs(off_diagonal.imag))),
        )
        correlations[(first, second)] = values
        if first != second:
            correlations[(second, first)] = values.T
    if assume_real_wavefunction:
        zeros = np.zeros((L, L), dtype=np.complex128)
        for first, second in (
            ("Sigmax", "Sigmay"),
            ("Sigmay", "Sigmax"),
            ("Sigmay", "Sigmaz"),
            ("Sigmaz", "Sigmay"),
        ):
            correlations[(first, second)] = zeros
    if maximum_correlation_imaginary_part > numerical_tolerance:
        raise ValueError(
            "Hermitian two-site Pauli expectations have a large imaginary part"
        )

    one_point_norm = np.sum(one_point**2, axis=0)
    single_site_purity = _bounded_purity(
        0.5 * (1.0 + one_point_norm),
        tolerance=numerical_tolerance,
    )
    single_site_entropy = -np.log(single_site_purity)

    correlation_norm = np.zeros((L, L), dtype=float)
    for first in PAULI_OPERATORS:
        for second in PAULI_OPERATORS:
            correlation_norm += correlations[(first, second)].real ** 2
    two_site_purity = 0.25 * (
        1.0
        + one_point_norm[:, None]
        + one_point_norm[None, :]
        + correlation_norm
    )
    mask = ~np.eye(L, dtype=bool)
    bounded_off_diagonal = _bounded_purity(
        two_site_purity[mask],
        tolerance=numerical_tolerance,
    )
    two_site_purity = np.array(two_site_purity, copy=True)
    two_site_purity[mask] = bounded_off_diagonal

    information = np.zeros((L, L), dtype=float)
    information[mask] = np.log(
        two_site_purity[mask]
        / (single_site_purity[:, None] * single_site_purity[None, :])[mask]
    )
    information = 0.5 * (information + information.T)
    np.fill_diagonal(information, 0.0)
    if np.any(~np.isfinite(information)):
        raise ValueError("Rényi-2 information contains non-finite values")

    return {
        "R2I": information,
        "single_site_purity": single_site_purity,
        "single_site_entropy": single_site_entropy,
        "maximum_correlation_imaginary_part": (
            maximum_correlation_imaginary_part
        ),
    }
