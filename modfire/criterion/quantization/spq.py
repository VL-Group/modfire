import logging

import torch
from torch import nn
import torch.nn.functional as F

from modfire import Consts
import modfire.train.hooks

from ..utils import pariwiseAffinity, pairwiseInnerProduct, CriterionRegistry

logger = logging.getLogger(Consts.Root)

@CriterionRegistry.register
class SPQ(nn.Module, modfire.train.hooks.StepStartHook):
    """
        Young Kyun Jang, Nam Ik Cho: Self-supervised Product Quantization for Deep Unsupervised Image Retrieval. ICCV 2021: 12065-12074
    """
    def __init__(self, d: int, temperature: float, tContrastive: float):
        super().__init__()
        self.d = d
        self.temperature = temperature
        self.tContrastive = tContrastive

    def stepStart(self, step: int, epoch: int, trainer, *_, **__):
        return { "temperature": self.temperature }

    def forward(self, *, x: torch.Tensor, q: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, **__):
        logits = pairwiseInnerProduct(x, q)
        labels = torch.cat((y, y))
        if torch.any(labels.sum(-1) > (1 + Consts.Eps)):
            # multi-label
            ceLoss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            # single-label, supports soft labels
            ceLoss = F.cross_entropy(logits, labels)

        n, d = x.shape
        c = len(self._centers)
        # [n, C, d]
        centerLoss = F.mse_loss(x[:, None, ...].expand(n, c, d), self._centers.expand(n, c, d), reduction="none") + F.mse_loss(q[:, None, ...].expand(n, c, d), self._centers.expand(n, c, d), reduction="none")
        # mask loss that not on label
        centerLoss = (centerLoss * y[..., None]).mean()

        # [n, m, k] -> [m, k] -> 1
        giniBatch = ((logits.softmax(-1).sum(0) / len(logits)) ** 2).mean()
        giniSample = -((logits.softmax(-1) ** 2).sum(-1)).mean()

        return ceLoss + centerLoss + 1e-3 * giniBatch + 1e-3 * giniSample, { "loss": ceLoss + centerLoss, "ceLoss": ceLoss, "centerLoss": centerLoss, "giniBatch": giniBatch, "giniSample": giniSample }
