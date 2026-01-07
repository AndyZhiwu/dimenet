import torch
import torch.nn as nn


class GlorotOrthogonal:
    """
    Generate a weight matrix with variance according to Glorot initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """

    def __init__(self, scale=2.0, seed=None):
        self.scale = scale
        self.seed = seed

    def __call__(self, tensor):
        assert len(tensor.shape) == 2
        if self.seed is not None:
            torch.manual_seed(self.seed)
        nn.init.orthogonal_(tensor)
        tensor.data *= torch.sqrt(torch.tensor(self.scale / ((tensor.shape[0] + tensor.shape[1]) * torch.var(tensor))))
        return tensor
