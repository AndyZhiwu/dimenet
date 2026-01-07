import torch
import torch.nn as nn

from ..initializers import GlorotOrthogonal


class OutputBlock(nn.Module):
    def __init__(self, emb_size, num_dense, num_targets=12,
                 activation=None, output_init='zeros'):
        super().__init__()
        self.activation = activation
        weight_init = GlorotOrthogonal()

        self.dense_rbf = nn.Linear(emb_size, emb_size, bias=False)
        weight_init(self.dense_rbf.weight)
        
        self.dense_layers = nn.ModuleList()
        for i in range(num_dense):
            dense = nn.Linear(emb_size, emb_size, bias=True)
            weight_init(dense.weight)
            self.dense_layers.append(dense)
            
        self.dense_final = nn.Linear(emb_size, num_targets, bias=False)
        if output_init == 'GlorotOrthogonal':
            weight_init(self.dense_final.weight)
        elif output_init == 'zeros':
            nn.init.zeros_(self.dense_final.weight)

    def forward(self, inputs):
        x, rbf, idnb_i, n_atoms = inputs

        g = self.dense_rbf(rbf)
        x = g * x
        x = torch.zeros(n_atoms, x.shape[-1], dtype=x.dtype, device=x.device).index_add_(0, idnb_i, x)

        for layer in self.dense_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        x = self.dense_final(x)
        return x
