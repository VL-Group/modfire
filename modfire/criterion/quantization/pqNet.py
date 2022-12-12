import logging

import torch
from torch import nn
import torch.nn.functional as F

from modfire import Consts
import modfire.train.hooks

from ..utils import pariwiseAffinity, pairwiseInnerProduct, CriterionRegistry

logger = logging.getLogger(Consts.Root)

@CriterionRegistry.register
class PQNet(nn.Module, modfire.train.hooks.StepStartHook):
    """
        Tan Yu, Junsong Yuan, Chen Fang, Hailin Jin: Product Quantization Network for Fast Image Retrieval. ECCV (1) 2018: 191-206
    """
    def __init__(self, mode: str, d: int, numClasses: int, temperature: float):
        super().__init__()
        if mode == "triplet":
            self._calcLoss = self._tripletImpl
            logger.debug("Use triplet PQNet.")
        elif mode == "ce":
            self._calcLoss = self._ceImpl
            self._finalLayer = nn.Linear(d, numClasses)
            logger.debug("Use cross-entropy PQNet.")
        else:
            raise ValueError(f"Mode is expected to be one of [\"ce\", \"triplet\"], but got: `{mode}`.")

        self.temperature = temperature

    def stepStart(self, step: int, epoch: int, trainer, *_, **__):
        return { "temperature": self.temperature }

    def _tripletImpl(self, x: torch.Tensor, q: torch.Tensor, y: torch.Tensor):
        # https://gist.github.com/rwightman/fff86a015efddcba8b3c8008167ea705
        # pairwise distances, since normalized, we directly use inner product
        dist = -pairwiseInnerProduct(x, q)

        affinity = pariwiseAffinity(y)

        # find the hardest positive and negative
        affinity = affinity
        notAffinity = ~affinity
        affinity[torch.eye(len(x), device=x.device, dtype=torch.uint8)] = 0
        if True:
            # weighted sample pos and negative to avoid outliers causing collapse
            posw = (dist + 1e-12) * affinity.float()
            posi = torch.multinomial(posw, 1)
            positiveDis = dist.gather(0, posi.view(1, -1))
            # There is likely a much better way of sampling negatives in proportion their difficulty, based on distance
            # this was a quick hack that ended up working better for some datasets than hard negative
            negw = (1 / (dist + 1e-12)) * notAffinity.float()
            negi = torch.multinomial(negw, 1)
            negativeDis = dist.gather(0, negi.view(1, -1))
        else:
            # hard negative
            ninf = torch.ones_like(dist) * float('-inf')
            positiveDis = torch.max(dist * affinity.float(), dim=1)[0]
            nindex = torch.max(torch.where(notAffinity, -dist, ninf), dim=1)[1]
            negativeDis = dist.gather(0, nindex.unsqueeze(0))

        # calc loss
        diff = positiveDis - negativeDis
        # Use softplus based on the original paper
        diff = F.softplus(diff)
        # diff = torch.clamp(diff + self.margin, min=0.)
        loss = diff.mean()

        return loss

    def _ceImpl(self, x: torch.Tensor, q: torch.Tensor, y: torch.Tensor):
        logits = torch.cat((self._finalLayer(x), self._finalLayer(q)))
        labels = torch.cat((y, y))
        if torch.any(labels.sum(-1) > (1 + Consts.Eps)):
            # multi-label
            return F.binary_cross_entropy_with_logits(logits, labels)
        else:
            # single-label, supports soft labels
            return F.cross_entropy(logits, labels)

    def forward(self, *, x: torch.Tensor, q: torch.Tensor, y: torch.Tensor, **__):
        loss = self._calcLoss(x, q, y)
        return loss, { "loss": loss }
