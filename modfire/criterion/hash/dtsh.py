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
    def __init__(self, alpha: float = 5., _lambda: float = 1.):
        super().__init__()
        self.alpha = alpha
        self._lambda = _lambda

    def forward(self, *, b: torch.Tensor, y: torch.Tensor, **_):
        # [N, N]
        affinity = pariwiseAffinity(y).float()
        similarity = pairwiseInnerProduct(b) * 0.5

        count = 0
        loss1 = 0
        for row in range(len(affinity)):
            # if has positive pairs and negative pairs
            if affinity[row].sum() != 0 and (~affinity[row]).sum() != 0:
                count += 1
                theta_positive = similarity[row][affinity[row]]
                theta_negative = similarity[row][~affinity[row]]
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - self.alpha).clamp(-100, 50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = ((b - b.sign()) ** 2).mean()

        return loss1 + self._lambda * loss2, {"loss1": loss1, "loss2": loss2}
