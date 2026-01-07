import torch
import torch.nn.functional as F


def swish(x):
    """
    Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """
    return x * torch.sigmoid(x)


def shifted_softplus(x):
    return F.softplus(x) - torch.log(torch.tensor(2.0))
