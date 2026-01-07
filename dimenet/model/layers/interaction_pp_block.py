import torch
import torch.nn as nn

from .residual_layer import ResidualLayer
from ..initializers import GlorotOrthogonal


class InteractionPPBlock(nn.Module):
    def __init__(self, emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip,
                 num_radial, num_spherical, activation=None):
        super().__init__()
        self.activation = activation
        weight_init = GlorotOrthogonal()

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.dense_rbf2 = nn.Linear(basis_emb_size, emb_size, bias=False)
        self.dense_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.dense_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        weight_init(self.dense_rbf1.weight)
        weight_init(self.dense_rbf2.weight)
        weight_init(self.dense_sbf1.weight)
        weight_init(self.dense_sbf2.weight)

        # Dense transformations of input messages
        self.dense_ji = nn.Linear(emb_size, emb_size, bias=True)
        self.dense_kj = nn.Linear(emb_size, emb_size, bias=True)
        weight_init(self.dense_ji.weight)
        weight_init(self.dense_kj.weight)

        # Embedding projections for interaction triplets
        self.down_projection = nn.Linear(emb_size, int_emb_size, bias=False)
        self.up_projection = nn.Linear(int_emb_size, emb_size, bias=False)
        weight_init(self.down_projection.weight)
        weight_init(self.up_projection.weight)

        # Residual layers before skip connection
        self.layers_before_skip = nn.ModuleList()
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=weight_init))
        self.final_before_skip = nn.Linear(emb_size, emb_size, bias=True)
        weight_init(self.final_before_skip.weight)

        # Residual layers after skip connection
        self.layers_after_skip = nn.ModuleList()
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=weight_init))

    def forward(self, inputs):
        x, rbf, sbf, id_expand_kj, id_reduce_ji = inputs
        num_interactions = x.shape[0]

        # Initial transformation
        x_ji = self.dense_ji(x)
        if self.activation is not None:
            x_ji = self.activation(x_ji)
        x_kj = self.dense_kj(x)
        if self.activation is not None:
            x_kj = self.activation(x_kj)

        # Transform via Bessel basis
        rbf = self.dense_rbf1(rbf)
        rbf = self.dense_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings
        x_kj = self.down_projection(x_kj)
        if self.activation is not None:
            x_kj = self.activation(x_kj)
        x_kj = x_kj[id_expand_kj]

        # Transform via 2D spherical basis
        sbf = self.dense_sbf1(sbf)
        sbf = self.dense_sbf2(sbf)
        x_kj = x_kj * sbf

        # Aggregate interactions and up-project embeddings
        x_kj = torch.zeros(num_interactions, x_kj.shape[-1], dtype=x_kj.dtype, device=x_kj.device).index_add_(0, id_reduce_ji, x_kj)
        x_kj = self.up_projection(x_kj)
        if self.activation is not None:
            x_kj = self.activation(x_kj)

        # Transformations before skip connection
        x2 = x_ji + x_kj
        for layer in self.layers_before_skip:
            x2 = layer(x2)
        x2 = self.final_before_skip(x2)
        if self.activation is not None:
            x2 = self.activation(x2)

        # Skip connection
        x = x + x2

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            x = layer(x)
        return x
