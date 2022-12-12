import logging
from typing import Any

from scipy.linalg import hadamard
import torch
from torch import nn
import torch.nn.functional as F

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseHamming, CriterionRegistry

logger = logging.getLogger(Consts.Root)


class ADSH(nn.Module):
    """
        Qing-Yuan Jiang, Wu-Jun Li: Asymmetric Deep Supervised Hashing. AAAI 2018: 3342-3349
    """
    databaseLabel: torch.Tensor
    U: torch.Tensor
    def __init__(self, sampleSize: int, bits: int):
        super().__init__()
        raise NotImplementedError("Method not implemented.")
        self.register_buffer("U", torch.zeros((sampleSize, bits)))

    def calcSim(self, sampledLabel: torch.Tensor, databaseLabel: torch.Tensor):
        S = ((sampledLabel @ databaseLabel.t()) > 0).float()
        # soft constraint
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r
        return S

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError("Method not implemented.")
        S = self.calcSim(y, self.databaseLabel)
        softSigned = x.tanh()
        # random sample
        randIdx = torch.randperm(len(self.U))[:len(x)]
        self.U.data[randIdx].copy_(softSigned)
