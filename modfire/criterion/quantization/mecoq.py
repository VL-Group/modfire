import logging
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseCosine, CriterionRegistry

logger = logging.getLogger(Consts.Root)

@CriterionRegistry.register
class MeCoQ(nn.Module, modfire.train.hooks.StepStartHook, modfire.train.hooks.AfterRunHook):
    """
        Jinpeng Wang, Ziyun Zeng, Bin Chen, Tao Dai, Shu-Tao Xia: Contrastive Quantization with Code Memory for Unsupervised Image Retrieval. AAAI 2022: 2468-2476
    """
    def __init__(self, d: int, temperature: float, _lambda: float, gamma: float, queueSize: int, posPrior: float):
        super().__init__()
        self.d = d
        self.temperature = temperature
        self.queue = torch.multiprocessing.Queue(maxsize=queueSize)
        self.posPrior = posPrior
        self._lambda = _lambda
        self.gamma = gamma

    def afterRun(self, step: int, epoch: int, trainer, *args, logger, **kwds):
        self.queue.close()

    def _simCLRQueue(self, q: torch.Tensor, y: torch.Tensor, queuedFeatures: torch.Tensor):
        # [N2]
        thatQ = torch.cat([q, queuedFeatures], 0)
        batchSize = len(q)
        # [N, N2]
        # [Za, Zb] vs [Za, Zb]
        logits = pairwiseCosine(q, thatQ) / self.tContrastive
        # [N, N2]
        mask = torch.cat((~torch.eye(len(logits), dtype=torch.bool, device=logits.device), torch.ones(batchSize, len(queuedFeatures), dtype=torch.bool, device=logits.device)), -1)
        # erase diagnoal
        # [N, N2 - 1]
        logits = logits[mask].reshape(batchSize, -1)
        # find another view as label
        # [N, N2]
        affinity = torch.cat((y[..., None] == y, torch.ones(batchSize, len(queuedFeatures), dtype=torch.bool, device=logits.device)), -1)
        # [N, N2 - 1]
        affinity = affinity[mask].reshape(batchSize, -1)

        # [N, N2-1]
        probs = logits.softmax(-1)

        # [N, 1]
        posProbs = probs[affinity > 0].reshape(batchSize, -1)
        # [N, N2-2]
        negProbs = probs[affinity < 0.5].reshape(batchSize, -1)

        # debias
        N = batchSize - 2
        Ng = torch.clamp((-self.posPrior * N * posProbs + negProbs.sum(dim=-1)) / (1 - self.posPrior), min=math.exp(N * (-1 / self.temperature)))

        loss = (- torch.log(posProbs / (negProbs + Ng))).mean()
        return loss


    def _simCLR(self, q: torch.Tensor, y: torch.Tensor, queuedFeatures: torch.Tensor):
        if queuedFeatures is not None:
            return self._simCLRQueue(q, y, queuedFeatures)

        batchSize = len(q)
        # [N, N]
        # [Za, Zb] vs [Za, Zb]
        logits = pairwiseCosine(q) / self.tContrastive
        mask = ~torch.eye(batchSize, dtype=torch.bool, device=logits.device)
        # erase diagnoal
        # [N, N - 1]
        logits = logits[mask].reshape(batchSize, -1)

        # find another view as label
        # [N, N]
        affinity = y[..., None] == y
        # [N, N - 1]
        affinity = affinity[mask].reshape(batchSize, -1)

        # [N, N-1]
        probs = logits.softmax(-1)

        # [N, 1]
        posProbs = probs[affinity > 0].reshape(batchSize, -1)
        # [N, N-2]
        negProbs = probs[affinity < 0.5].reshape(batchSize, -1)


        # debias
        N = batchSize - 2
        Ng = torch.clamp((-self.posPrior * N * posProbs + negProbs.sum(dim=-1)) / (1 - self.posPrior), min=math.exp(N * (-1 / self.temperature)))

        loss = (- torch.log(posProbs / (negProbs + Ng))).mean()
        return loss

    def _updateQueue(self, q: torch.Tensor):
        if self.queue.full():
            self.queue.get()
        self.queue.put(q)

    def _getQueuedFeatures(self):
        return torch.cat(self.queue.queue)

    def forward(self, *, x: torch.Tensor, q: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, codebook: torch.Tensor, **__):
        contrastive = self._simCLR(q, y, self._getQueuedFeatures())

        logitsReg = (- logits * logits.log()).sum(dim=-1).mean()
        codebookReg = torch.einsum('mkd,mjd->mkj', codebook, codebook).mean()

        return loss, { }
