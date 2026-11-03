"""Fit and envelope constants for the local-error bound."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundSummary:
    C_env: float
    C_LS: float
    ratios: np.ndarray
    used_mask: np.ndarray


def estimate_bound_constants(
    sqrt_delta_max: np.ndarray,
    epsilon_rho: np.ndarray,
    tol: float = 1e-14,
) -> BoundSummary:
    x = np.asarray(sqrt_delta_max, dtype=float)
    y = np.asarray(epsilon_rho, dtype=float)
    if x.shape != y.shape:
        raise ValueError("sqrt_delta_max and epsilon_rho must have the same shape")

    used = x > tol
    ratios = np.full_like(x, np.nan, dtype=float)
    ratios[used] = y[used] / x[used]

    C_env = float(np.nanmax(ratios)) if np.any(used) else 0.0
    denom = float(np.sum(x[used] ** 2))
    C_LS = float(np.sum(x[used] * y[used]) / denom) if denom > 0.0 else 0.0
    return BoundSummary(C_env=C_env, C_LS=C_LS, ratios=ratios, used_mask=used)
