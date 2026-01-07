import torch
import torch.nn as nn

from .residual_layer import ResidualLayer
from ..initializers import GlorotOrthogonal


class InteractionBlock(nn.Module):
    def __init__(self, emb_size, num_bilinear, num_before_skip, num_after_skip,
                 num_radial, num_spherical, activation=None):
        super().__init__()
        self.emb_size = emb_size
        self.num_bilinear = num_bilinear
        self.activation = activation
        weight_init = GlorotOrthogonal()

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_sbf = nn.Linear(num_spherical * num_radial, num_bilinear, bias=False)
        weight_init(self.dense_rbf.weight)
        weight_init(self.dense_sbf.weight)

        # Dense transformations of input messages
        self.dense_ji = nn.Linear(emb_size, emb_size, bias=True)
        self.dense_kj = nn.Linear(emb_size, emb_size, bias=True)
        weight_init(self.dense_ji.weight)
        weight_init(self.dense_kj.weight)

        # Bilinear layer
        self.W_bilin = nn.Parameter(torch.empty(emb_size, num_bilinear, emb_size))
        nn.init.normal_(self.W_bilin, mean=0.0, std=2 / emb_size)

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
        g = self.dense_rbf(rbf)
        x_kj = x_kj * g

        # Transform via spherical basis
        sbf = self.dense_sbf(sbf)
        x_kj = x_kj[id_expand_kj]
        # Apply bilinear layer to interactions and basis function activation
        x_kj = torch.einsum("wj,wl,ijl->wi", sbf, x_kj, self.W_bilin)
        x_kj = torch.zeros(num_interactions, x_kj.shape[-1], dtype=x_kj.dtype, device=x_kj.device).index_add_(0, id_reduce_ji, x_kj)

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
