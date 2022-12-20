import logging

import torch
from torch import nn
import torch.nn.functional as F

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseCosine, CriterionRegistry

logger = logging.getLogger(Consts.Root)

@CriterionRegistry.register
class SPQ(nn.Module, modfire.train.hooks.StepStartHook):
    """
        Young Kyun Jang, Nam Ik Cho: Self-supervised Product Quantization for Deep Unsupervised Image Retrieval. ICCV 2021: 12065-12074
    """
    def __init__(self, d: int, temperature: float):
        super().__init__()
        self.d = d
        self.temperature = temperature

    def forward(self, *, x: torch.Tensor, q: torch.Tensor, y: torch.Tensor, **__):
        # [N, N]
        # [Xa, Xb] vs [Za, Zb]
        logits = pairwiseCosine(x, q) / self.temperature
        mask = ~torch.eye(len(logits), dtype=torch.bool, device=logits.device)
        # erase diagnoal
        # [N, N - 1]
        logits = logits[mask].reshape(len(x), -1)

        # find another view as label
        # [N, N]
        affinity = y[..., None] == y
        # [N, N - 1]
        affinity = affinity[mask].reshape(len(x), -1)

        loss = F.cross_entropy(logits, affinity.float().argmax(-1))

        return loss, { }
