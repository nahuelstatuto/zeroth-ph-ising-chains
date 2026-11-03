"""Pairwise von Neumann and Renyi-2 information from finite qubit MPS."""

from __future__ import annotations

from typing import Any

import numpy as np


PAULI_OPERATORS = ("Sigmax", "Sigmay", "Sigmaz")
_IDENTITY = np.eye(2, dtype=np.complex128)
_PAULI_MATRICES = (
    np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
)
_LEFT_BASIS = tuple(np.kron(matrix, _IDENTITY) for matrix in _PAULI_MATRICES)
_RIGHT_BASIS = tuple(np.kron(_IDENTITY, matrix) for matrix in _PAULI_MATRICES)
_PAIR_BASIS = tuple(
    tuple(np.kron(first, second) for second in _PAULI_MATRICES)
    for first in _PAULI_MATRICES
)


def _entropy_from_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    tolerance: float,
) -> float:
    values = np.asarray(eigenvalues, dtype=float)
    if np.any(~np.isfinite(values)) or np.min(values) < -tolerance:
        raise ValueError("reduced density matrix is not positive semidefinite")
    values = np.clip(values, 0.0, None)
    normalization = float(np.sum(values))
    if not np.isclose(normalization, 1.0, rtol=0.0, atol=10.0 * tolerance):
        raise ValueError("reduced density matrix has invalid trace")
    values = values / normalization
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log(values[positive])))


def _collect_pauli_expectations(
    state: Any,
    *,
    numerical_tolerance: float,
    assume_real_wavefunction: bool,
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], float]:
    L = int(state.L)
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

    correlations: dict[tuple[str, str], np.ndarray] = {}
    maximum_imaginary_part = 0.0
    off_diagonal_mask = ~np.eye(L, dtype=bool)
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
        maximum_imaginary_part = max(
            maximum_imaginary_part,
            float(np.max(np.abs(values[off_diagonal_mask].imag))),
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
    if maximum_imaginary_part > numerical_tolerance:
        raise ValueError(
            "Hermitian two-site Pauli expectations have a large imaginary part"
        )
    return one_point, correlations, maximum_imaginary_part


def pair_information_matrices(
    state: Any,
    *,
    numerical_tolerance: float = 1e-10,
    assume_real_wavefunction: bool = False,
) -> dict[str, np.ndarray | float]:
    r"""Compute MI and Renyi-2 information for every distinct pair of sites.

    The von Neumann mutual information and Renyi-2 information are

    .. math::

        I(i:j) = S(\rho_i) + S(\rho_j) - S(\rho_{ij}),

        I_2(i:j) = S_2(\rho_i) + S_2(\rho_j) - S_2(\rho_{ij}).

    Natural logarithms are used. Complete one- and two-site Pauli expectation
    values reconstruct each two-qubit reduced density matrix. Both returned
    matrices use a zero diagonal as a pair-observable convention.
    """

    L = int(getattr(state, "L", 0))
    if L < 2 or getattr(state, "bc", None) != "finite":
        raise ValueError("pair information requires a finite MPS with L >= 2")
    if any(int(getattr(site, "dim", 0)) != 2 for site in state.sites):
        raise ValueError("Pauli reconstruction requires qubit sites")
    if numerical_tolerance <= 0.0:
        raise ValueError("numerical_tolerance must be positive")

    one_point, correlations, maximum_imaginary_part = (
        _collect_pauli_expectations(
            state,
            numerical_tolerance=numerical_tolerance,
            assume_real_wavefunction=assume_real_wavefunction,
        )
    )
    correlation_tensor = np.empty((3, 3, L, L), dtype=float)
    for first_index, first in enumerate(PAULI_OPERATORS):
        for second_index, second in enumerate(PAULI_OPERATORS):
            correlation_tensor[first_index, second_index] = correlations[
                (first, second)
            ].real

    bloch_norm = np.linalg.norm(one_point, axis=0)
    if np.any(bloch_norm > 1.0 + numerical_tolerance):
        raise ValueError("one-site Bloch vector lies outside the unit ball")
    bloch_norm = np.clip(bloch_norm, 0.0, 1.0)
    single_site_eigenvalues = np.stack(
        [0.5 * (1.0 + bloch_norm), 0.5 * (1.0 - bloch_norm)],
        axis=1,
    )
    single_site_von_neumann_entropy = np.asarray(
        [
            _entropy_from_eigenvalues(
                values,
                tolerance=numerical_tolerance,
            )
            for values in single_site_eigenvalues
        ],
        dtype=float,
    )
    single_site_purity = 0.5 * (1.0 + bloch_norm**2)
    single_site_renyi2_entropy = -np.log(single_site_purity)

    mutual_information = np.zeros((L, L), dtype=float)
    renyi2_information = np.zeros((L, L), dtype=float)
    minimum_two_site_eigenvalue = np.inf
    maximum_two_site_trace_error = 0.0
    identity_pair = np.eye(4, dtype=np.complex128)

    for i in range(L):
        for j in range(i + 1, L):
            rho = np.array(identity_pair, copy=True)
            for axis in range(3):
                rho += one_point[axis, i] * _LEFT_BASIS[axis]
                rho += one_point[axis, j] * _RIGHT_BASIS[axis]
                for second_axis in range(3):
                    rho += (
                        correlation_tensor[axis, second_axis, i, j]
                        * _PAIR_BASIS[axis][second_axis]
                    )
            rho *= 0.25
            rho = 0.5 * (rho + rho.conj().T)
            trace_error = abs(float(np.trace(rho).real) - 1.0)
            maximum_two_site_trace_error = max(
                maximum_two_site_trace_error,
                trace_error,
            )
            eigenvalues = np.linalg.eigvalsh(rho)
            minimum_two_site_eigenvalue = min(
                minimum_two_site_eigenvalue,
                float(np.min(eigenvalues)),
            )
            pair_entropy = _entropy_from_eigenvalues(
                eigenvalues,
                tolerance=numerical_tolerance,
            )
            pair_purity = float(np.trace(rho @ rho).real)
            if (
                pair_purity < -numerical_tolerance
                or pair_purity > 1.0 + numerical_tolerance
            ):
                raise ValueError("two-site purity lies outside [0,1]")
            pair_purity = float(
                np.clip(pair_purity, np.finfo(float).tiny, 1.0)
            )

            mi = (
                single_site_von_neumann_entropy[i]
                + single_site_von_neumann_entropy[j]
                - pair_entropy
            )
            if mi < -10.0 * numerical_tolerance:
                raise ValueError("von Neumann mutual information is negative")
            if abs(mi) <= 10.0 * numerical_tolerance:
                mi = 0.0
            r2i = np.log(
                pair_purity
                / (single_site_purity[i] * single_site_purity[j])
            )
            mutual_information[i, j] = mutual_information[j, i] = mi
            renyi2_information[i, j] = renyi2_information[j, i] = r2i

    return {
        "MI": mutual_information,
        "R2I": renyi2_information,
        "single_site_von_neumann_entropy": (
            single_site_von_neumann_entropy
        ),
        "single_site_renyi2_entropy": single_site_renyi2_entropy,
        "maximum_correlation_imaginary_part": maximum_imaginary_part,
        "minimum_two_site_eigenvalue": minimum_two_site_eigenvalue,
        "maximum_two_site_trace_error": maximum_two_site_trace_error,
    }
