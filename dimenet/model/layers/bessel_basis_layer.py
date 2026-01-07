import numpy as np
import torch
import torch.nn as nn

from .envelope import Envelope


class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent=5):
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = torch.tensor(1 / cutoff, dtype=torch.float32)
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        freq_init = torch.tensor(np.pi * np.arange(1, num_radial + 1, dtype=np.float32), dtype=torch.float32)
        self.frequencies = nn.Parameter(freq_init)

    def forward(self, inputs):
        d_scaled = inputs * self.inv_cutoff

        # Necessary for proper broadcasting behaviour
        d_scaled = d_scaled.unsqueeze(-1)

        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)
