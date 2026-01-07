import torch
import torch.nn as nn


class ResidualLayer(nn.Module):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super().__init__()
        self.dense_1 = nn.Linear(units, units, bias=use_bias)
        self.dense_2 = nn.Linear(units, units, bias=use_bias)
        self.activation = activation
        
        # Initialize weights
        if kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.dense_1.weight)
            nn.init.xavier_uniform_(self.dense_2.weight)
        elif hasattr(kernel_initializer, '__call__'):
            kernel_initializer(self.dense_1.weight)
            kernel_initializer(self.dense_2.weight)
            
        if use_bias and bias_initializer == 'zeros':
            nn.init.zeros_(self.dense_1.bias)
            nn.init.zeros_(self.dense_2.bias)

    def forward(self, inputs):
        x1 = self.dense_1(inputs)
        if self.activation is not None:
            x1 = self.activation(x1)
        x2 = self.dense_2(x1)
        if self.activation is not None:
            x2 = self.activation(x2)
        return inputs + x2
