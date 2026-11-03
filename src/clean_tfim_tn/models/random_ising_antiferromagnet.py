"""TeNPy MPO model for the reentrant random Ising antiferromagnet."""

from __future__ import annotations

import numpy as np

from clean_tfim_tn.disorder import normalize_bc


def _parameter_array(values, *, name: str, length: int, positive: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,) or array.ndim != 1:
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive")
    if not positive and np.any(array < 0.0):
        raise ValueError(f"{name} values must be non-negative")
    return array


class RandomIsingAntiferromagnet:
    """Factory for a finite TeNPy MPO using direct Pauli operators."""

    @staticmethod
    def build(model_params):
        try:
            from tenpy.models.model import CouplingMPOModel
            from tenpy.networks.site import SpinHalfSite
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise ImportError(
                "TeNPy is required for RandomIsingAntiferromagnet; install physics-tenpy."
            ) from exc

        class _RandomIsingAntiferromagnet(CouplingMPOModel):
            default_lattice = "Chain"

            def init_sites(self, params):
                return SpinHalfSite(conserve=None)

            def init_terms(self, params):
                raw_L = params.get("L", None)
                raw_bc = params.get("bc", "open")
                raw_J = params.get("J", None)
                raw_Gamma = params.get("Gamma", None)
                raw_h = params.get("h", 0.0)
                if raw_L is None:
                    raise ValueError("L is required")
                L = int(raw_L)
                if L < 2:
                    raise ValueError("L must be at least 2")
                bc = normalize_bc(raw_bc)
                n_bonds = L - 1 if bc == "open" else L
                J = _parameter_array(
                    raw_J, name="J", length=n_bonds, positive=True
                )
                Gamma = _parameter_array(
                    raw_Gamma, name="Gamma", length=L, positive=False
                )
                h = float(raw_h)
                if not np.isfinite(h) or h < 0.0:
                    raise ValueError("h must be a finite non-negative number")

                for i in range(L - 1):
                    self.add_coupling_term(
                        float(J[i]), i, i + 1, "Sigmaz", "Sigmaz", category="J_ZZ"
                    )
                if bc == "periodic":
                    self.add_coupling_term(
                        float(J[-1]), 0, L - 1, "Sigmaz", "Sigmaz", category="J_ZZ"
                    )

                for i in range(L):
                    self.add_onsite_term(
                        -float(Gamma[i]), i, "Sigmax", category="Gamma_X"
                    )
                    if h != 0.0:
                        self.add_onsite_term(-h, i, "Sigmaz", category="h_Z")

                self.physical_bc = bc
                self.J_values = J.copy()
                self.Gamma_values = Gamma.copy()
                self.h_value = h

        params = dict(model_params)
        params["bc_MPS"] = "finite"
        # PBC is represented explicitly by the long-range first-last term.
        params["bc_x"] = "open"
        return _RandomIsingAntiferromagnet(params)
