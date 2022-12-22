import logging
from typing import Any

from scipy.linalg import hadamard
import torch
from torch import nn
import torch.nn.functional as F

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseSquare, CriterionRegistry, pariwiseAffinity

logger = logging.getLogger(Consts.Root)

@CriterionRegistry.register
class DSH(nn.Module, modfire.train.hooks.BeforeRunHook):
    """
        Haomiao Liu, Ruiping Wang, Shiguang Shan, Xilin Chen: Deep Supervised Hashing for Fast Image Retrieval. CVPR 2016: 2064-2072
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, *, b: torch.Tensor, y: torch.Tensor, **_):
        # [N, N]
        dist = pairwiseSquare(b)
        affinity = pariwiseAffinity(y).float()
        loss1 = ((1 - affinity) / 2 * dist + affinity / 2 * (2 * b.shape[-1] - dist).clamp_min_(0)).mean()
        loss2 = (1 - b.abs()).abs().mean()

        return loss1 + self.alpha * loss2, {"loss1": loss1, "loss2": loss2}
