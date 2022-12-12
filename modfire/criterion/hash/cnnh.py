import logging
from typing import Any
from itertools import product
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import IterDataPipe

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseHamming, CriterionRegistry

logger = logging.getLogger(Consts.Root)


@CriterionRegistry.register
class CNNH(nn.Module, modfire.train.hooks.BeforeRunHook):
    """
        Rongkai Xia, Yan Pan, Hanjiang Lai, Cong Liu, Shuicheng Yan: Supervised Hashing for Image Retrieval via Image Representation Learning. AAAI 2014: 2156-2162
    """
    S: torch.Tensor
    H: torch.Tensor
    def __init__(self, bits: int, stageOneLoop: int):
        super().__init__()
        self.bits = bits
        self.stageOneLoop = stageOneLoop


    def beforeRun(self, step: int, epoch: int, trainer, *_, trainSet, logger):
        self.register_buffer("H", self.stageOne(trainSet, logger))

    def stageOne(self, trainSet, saver):
        trainsetLabels: torch.Tensor = trainSet.Labels
        trainSetSize = len(trainSet)
        similarityMatrix = (trainsetLabels @ trainsetLabels.t() > 0).float() * 2 - 1

        H = 2 * torch.rand(trainSetSize, self.bits) - 1
        L = H @ H.t() - self.bits * similarityMatrix
        permutation = list(product(range(trainSetSize), range(self.bits)))
        for t in range(self.stageOneLoop):
            H_temp = H.clone()
            L_temp = L.clone()
            random.shuffle(permutation)
            for i, j in permutation:
                # formula 7
                g_prime_Hij = 4 * L[i, :] @ H[:, j]
                g_prime_prime_Hij = 4 * (H[:, j].t() @ H[:, j] + H[i, j].pow(2) + L[i, i])
                # formula 6
                d = (-g_prime_Hij / g_prime_prime_Hij).clamp(min=-1 - H[i, j], max=1 - H[i, j])
                # formula 8
                L[i, :] = L[i, :] + d * H[:, j].t()
                L[:, i] = L[:, i] + d * H[:, j]
                L[i, i] = L[i, i] + d * d

                H[i, j] = H[i, j] + d

            if L.pow(2).mean() >= L_temp.pow(2).mean():
                H = H_temp
                L = L_temp
            saver.info("[CNNH stage 1][%d/%d] reconstruction loss:%.7f" % (t + 1, self.stageOneLoop, L.pow(2).mean().item()))
        return H.sign()


    def forward(self, x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor):
        loss = (x - self.H[idx]).pow(2).mean()
        return loss
