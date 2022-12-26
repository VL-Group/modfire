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
class DHN(nn.Module):
    """
        Han Zhu, Mingsheng Long, Jianmin Wang, Yue Cao: Deep Hashing Network for Efficient Similarity Retrieval. AAAI 2016: 2415-2421
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, *, b: torch.Tensor, y: torch.Tensor, **_):
        # [N, N]
        affinity = pariwiseAffinity(y).float()
        similarity = pairwiseInnerProduct(b) * 0.5

        likelihoodLoss = ((1 + (-(similarity.abs())).exp()).log() + similarity.clamp_min(0) - affinity * similarity).mean()
        quantizationLoss = (b.abs() - 1).cosh().log().mean()

        return likelihoodLoss + self.alpha * quantizationLoss, {"likelihoodLoss": likelihoodLoss, "quantizationLoss": quantizationLoss}
