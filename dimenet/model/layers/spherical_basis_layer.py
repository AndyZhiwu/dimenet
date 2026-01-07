import sympy as sym
import torch
import torch.nn as nn

from .basis_utils import bessel_basis, real_sph_harm
from .envelope import Envelope


class SphericalBasisLayer(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent=5):
        super().__init__()

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical

        self.inv_cutoff = torch.tensor(1 / cutoff, dtype=torch.float32)
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(num_spherical, num_radial)
        self.sph_harm_formulas = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to pytorch functions
        x = sym.symbols('x')
        theta = sym.symbols('theta')
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify([theta], self.sph_harm_formulas[i][0], 'numpy')(0)
                self.sph_funcs.append(lambda tensor: torch.zeros_like(tensor) + first_sph)
            else:
                # Create lambda with numpy backend and wrap with torch
                sph_func = sym.lambdify([theta], self.sph_harm_formulas[i][0], 'numpy')
                self.sph_funcs.append(lambda t, f=sph_func: torch.tensor(f(t.cpu().numpy()), dtype=t.dtype, device=t.device))
            for j in range(num_radial):
                bessel_func = sym.lambdify([x], self.bessel_formulas[i][j], 'numpy')
                self.bessel_funcs.append(lambda t, f=bessel_func: torch.tensor(f(t.cpu().numpy()), dtype=t.dtype, device=t.device))

    def forward(self, inputs):
        d, Angles, id_expand_kj = inputs

        d_scaled = d * self.inv_cutoff
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = torch.stack(rbf, dim=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff.unsqueeze(1) * rbf
        rbf_env = rbf_env[id_expand_kj]

        cbf = [f(Angles) for f in self.sph_funcs]
        cbf = torch.stack(cbf, dim=1)
        cbf = cbf.repeat_interleave(self.num_radial, dim=1)

        return rbf_env * cbf
