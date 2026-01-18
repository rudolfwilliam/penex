from abc import ABC

from torch import nn


class ForwardModule(nn.Module, ABC):
    """Base class for the forward modules of the classifiers."""
    def __init__(self, n_out, deep_supervision):
        super().__init__()
        self.n_out = n_out
        self.deep_supervision = deep_supervision
