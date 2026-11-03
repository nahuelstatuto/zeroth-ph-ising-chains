"""Clean Pauli ZZ+X transverse-field Ising chain in TeNPy conventions."""

from __future__ import annotations


class CleanZZXTFIM:
    """Factory wrapper for the clean TFIM TeNPy MPO model.

    The actual TeNPy base class is imported lazily so the ED-only tests can run
    in environments where TeNPy has not been installed yet.
    """

    @staticmethod
    def build(model_params):
        try:
            from tenpy.models.model import CouplingMPOModel
            from tenpy.networks.site import SpinHalfSite
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise ImportError(
                "TeNPy is required for CleanZZXTFIM.build(); install physics-tenpy."
            ) from exc

        class _CleanZZXTFIM(CouplingMPOModel):
            default_lattice = "Chain"

            def init_sites(self, params):
                conserve = params.get("conserve", None)
                if conserve not in {None, "parity"}:
                    raise ValueError("conserve must be None or 'parity'")
                return SpinHalfSite(conserve=conserve)

            def init_terms(self, params):
                J = float(params.get("J", 1.0))
                hx = float(params.get("hx", 0.5))
                conserve = params.get("conserve", None)
                if conserve == "parity":
                    # Hadamard-rotated representation of physical ZZ+X. This
                    # exposes the global physical-X symmetry as TeNPy's
                    # standard Z-parity charge.
                    self.add_coupling(J, 0, "Sigmax", 0, "Sigmax", 1)
                    self.add_onsite(-hx, 0, "Sigmaz")
                else:
                    self.add_coupling(J, 0, "Sigmaz", 0, "Sigmaz", 1)
                    self.add_onsite(-hx, 0, "Sigmax")

        return _CleanZZXTFIM(model_params)
