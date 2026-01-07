import numpy as np
import torch
import torch.nn as nn

from ..initializers import GlorotOrthogonal


class EmbeddingBlock(nn.Module):
    def __init__(self, emb_size, activation=None):
        super().__init__()
        self.emb_size = emb_size
        self.activation = activation

        # Atom embeddings: We go up to Pu (94). Use 95 dimensions because of 0-based indexing
        self.embeddings = nn.Parameter(torch.empty(95, emb_size))
        nn.init.uniform_(self.embeddings, -np.sqrt(3), np.sqrt(3))

        self.dense_rbf = nn.Linear(emb_size, emb_size, bias=True)
        self.dense = nn.Linear(emb_size * 3, emb_size, bias=True)
        
        # Initialize weights
        weight_init = GlorotOrthogonal()
        weight_init(self.dense_rbf.weight)
        weight_init(self.dense.weight)

    def forward(self, inputs):
        Z, rbf, idnb_i, idnb_j = inputs

        rbf = self.dense_rbf(rbf)
        if self.activation is not None:
            rbf = self.activation(rbf)

        Z_i = Z[idnb_i]
        Z_j = Z[idnb_j]

        x_i = self.embeddings[Z_i]
        x_j = self.embeddings[Z_j]

        x = torch.cat([x_i, x_j, rbf], dim=-1)
        x = self.dense(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
