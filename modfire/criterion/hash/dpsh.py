import logging
from typing import Any

from scipy.linalg import hadamard
import torch
from torch import nn
import torch.nn.functional as F

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseInnerProduct, CriterionRegistry, pariwiseAffinity

logger = logging.getLogger(Consts.Root)

@CriterionRegistry.register
class DPSH(nn.Module):
    """
        Wu-Jun Li, Sheng Wang, Wang-Cheng Kang: Feature Learning Based Deep Supervised Hashing with Pairwise Labels. IJCAI 2016: 1711-1717
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, *, b: torch.Tensor, y: torch.Tensor, **_):
        # [N, N]
        affinity = pariwiseAffinity(y).float()
        similarity = pairwiseInnerProduct(b) * 0.5

        likelihoodLoss = ((1 + (-(similarity.abs())).exp()).log() + similarity.clamp_min(0) - affinity * similarity).mean()
        quantizationLoss = ((b - b.sign()) ** 2).mean()

        return likelihoodLoss + self.alpha * quantizationLoss, {"likelihoodLoss": likelihoodLoss, "quantizationLoss": quantizationLoss}
