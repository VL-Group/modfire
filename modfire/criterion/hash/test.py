import logging

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli, kl_divergence

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseHamming, CriterionRegistry, pairwiseCosine, pariwiseAffinity

logger = logging.getLogger(Consts.Root)


@CriterionRegistry.register
class TestContrastive(nn.Module, modfire.train.hooks.EpochFinishHook, AllReduce):
    # NOTE: A very interesting thing:
    #       Even if we don't train the mapNet
    #       The Hashing performance is still very high.
    class _randomBitFlip(nn.Module):
        template: torch.BoolTensor
        def __init__(self, bits, numBitsToFlip):
            super().__init__()
            self.bits = bits
            self.numBitsToFlip = numBitsToFlip
            template = torch.tensor([False] * bits)
            template[:numBitsToFlip] = True
            template = template[torch.randperm(len(template))]
            self.register_buffer('template', template)

        @property
        def BitFlip(self) -> int:
            return self.numBitsToFlip

        @BitFlip.setter
        def BitFlip(self, numBitsToFlip):
            self.numBitsToFlip = numBitsToFlip
            template = torch.tensor([False] * self.bits)
            template[:numBitsToFlip] = True
            template = template[torch.randperm(len(template))]
            self.template.copy_(template.to(self.template.device))

        @torch.no_grad()
        def forward(self, x):
            # [N, D]
            rand = torch.rand_like(x, dtype=torch.float)
            # randperm row-by-row
            batchRandIdx = rand.argsort(-1)
            shuffledTemplate = self.template[batchRandIdx]
            # STE
            x[shuffledTemplate] *= -1
            return x

    class _mapNet(nn.Module):
        def __init__(self, bits):
            super().__init__()
            self.m = bits // 8
            ffnNet = lambda: nn.Sequential(
                nn.Linear(8, 256),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 256)
            )
            self._net = nn.ModuleList(ffnNet() for _ in range(bits // 8))
            self._bitFlip = TestContrastive._randomBitFlip(bits, int(bits // 16) ** 2)

        def forward(self, x, flip):
            if flip:
                x = self._bitFlip(x)
            # x = F.dropout(x, 0.1, inplace=True)
            predict = list()
            for i, split in enumerate(torch.chunk(x, self.m, -1)):
                predict.append(self._net[i](split))
            return torch.cat(predict, -1)

    multiplier: torch.LongTensor
    permIdx: torch.LongTensor

    def __init__(self, bits: int, gamma: float, temperature: float):
        super().__init__()
        self.bits = bits
        self.bitFlip = self._randomBitFlip(bits, int(bits // 16) ** 2)
        self.mapper = self._mapNet(bits)
        self.register_buffer("multiplier", (2 ** torch.arange(8)).long())
        self.register_buffer("permIdx", torch.randperm(bits))
        self.m = bits // 8
        self._worldSize = dist.get_world_size()
        self._rank = dist.get_rank()

    @property
    def BitFlip(self) -> int:
        return self.mapper._bitFlip.BitFlip

    @BitFlip.setter
    def BitFlip(self, numBitsToFlip: int):
        self.mapper._bitFlip.BitFlip = numBitsToFlip
        self.bitFlip.BitFlip = numBitsToFlip

    def reset(self):
        # reset permIdx
        self.permIdx.data.copy_(torch.randperm(self.m * 8, device=self.permIdx.device))
        # reset params
        # self.mapper.reset()

    def epochFinish(self, step: int, epoch: int, *_, logger, **__):
        logger.debug("Call `%s.epochFinish()`.", self._get_name())
        if True:
            logger.debug("Reset permutation index in `%s`.", self._get_name())
            self.reset()

    def splitFeatureInTwoViews(self, z: torch.Tensor, y: torch.Tensor):
        idx = torch.arange(len(z)).reshape(-1, 2)
        return z[idx[:, 0]], z[idx[:, 1]]
        # [?]
        uniqueLabels = torch.unique(y)
        lefts = list()
        rights = list()
        for l in uniqueLabels:
            matched = y == l
            matchedIdx = torch.nonzero(matched)
            left, right = matchedIdx[0], matchedIdx[1]
            lefts.append(left)
            rights.append(right)
        return z[torch.cat(lefts)], z[torch.cat(rights)]

    def mapping(self, left: torch.Tensor, right: torch.Tensor, **_):

        mapLoss = list()
        hitRate = list()

        resultDict = {}


        # X are permuted on last dim according to permIdx
        left = left[:, self.permIdx]
        right = right[:, self.permIdx]


        originalLeft = left.clone().detach()
        leftDec = self.mapper(left.detach(), True)
        leftSplit = torch.chunk(leftDec, self.m, -1)
        leftRawSplit = torch.chunk(self.bitFlip(originalLeft), self.m, -1)

        for subX, rawSubX in zip(leftSplit, leftRawSplit):
            binary = rawSubX > 0
            target = (self.multiplier * binary).sum(-1)

            reverse = self.bitFlip(rawSubX) <= 0
            reverse = (self.multiplier * reverse).sum(-1)

            mapLoss.append(F.cross_entropy(subX, target) - F.cross_entropy(subX, reverse, reduction='none').clamp_max(16).mean())
            hitRate.append((subX.argmax(-1) == target).float().mean())

        originalRight = right.clone().detach()
        rightDec = self.mapper(right.detach(), True)
        rightSplit = torch.chunk(rightDec, self.m, -1)
        rightRawSplit = torch.chunk(self.bitFlip(originalRight), self.m, -1)

        for subX, rawSubX in zip(rightSplit, rightRawSplit):
            binary = rawSubX > 0
            target = (self.multiplier * binary).sum(-1)

            reverse = self.bitFlip(rawSubX) <= 0
            reverse = (self.multiplier * reverse).sum(-1)

            mapLoss.append(F.cross_entropy(subX, target) - F.cross_entropy(subX, reverse, reduction='none').clamp_max(16).mean())
            hitRate.append((subX.argmax(-1) == target).float().mean())


        mapLoss = sum(mapLoss)

        resultDict.update({ "mapLoss": mapLoss, "hitRate": sum(hitRate) / self.m })

        return mapLoss, resultDict

    def contrastive(self, left: torch.Tensor, right: torch.Tensor):
        regLoss = list()
        swapLoss = list()
        rejectLoss = list()
        resultDict = dict()
        # X are permuted on last dim according to permIdx
        left = left[:, self.permIdx]
        right = right[:, self.permIdx]

        originalLeft = left.clone().detach()
        originalRight = right.clone().detach()

        leftDecGrad = self.mapper(left, False)
        # M * [N, 256]
        leftSplitted = torch.chunk(leftDecGrad, self.m, -1)
        leftRawSplit = torch.chunk(self.bitFlip(originalLeft), self.m, -1)

        rightDecGrad = self.mapper(right, False)
        # M * [N, 256]
        rightSplitted = torch.chunk(rightDecGrad, self.m, -1)
        rightRawSplit = torch.chunk(self.bitFlip(originalRight), self.m, -1)

        N = len(left)
        batchSize = 2 * len(left)

        mask = (~torch.eye(batchSize, dtype=torch.bool))

        upper, lower = mask.diagonal(N), mask.diagonal(-N)
        upper.copy_(False)
        lower.copy_(False)

        for i, (leftSub, rightSub, leftRawSub, rightRawSub) in enumerate(zip(leftSplitted, rightSplitted, leftRawSplit, rightRawSplit)):
            # [N]
            leftCode =  (self.multiplier * (leftRawSub > 0)).sum(-1)
            rightCode = (self.multiplier * (rightRawSub > 0)).sum(-1)

            # [2N]
            allCodes = torch.cat((leftCode, rightCode))
            # [2N, 256]
            allLogits = torch.cat((leftSub, rightSub))
            # [2N, 256]
            allLogits = -allLogits.log_softmax(-1)

            allCodesList = [torch.empty_like(allCodes) for _ in range(self._worldSize)]
            allLogitsList = [torch.empty_like(allLogits) for _ in range(self._worldSize)]

            ###################### Collect Across All GPUs ######################
            dist.all_gather(allCodesList, allCodes)

            # replace tensor slice of current group to the one that has gradients
            allLogitsList[self._rank] = allLogits

            # [2N * worldSize]
            allCodes = torch.cat(allCodesList)
            # [2N * worldSize, 256]
            allCodes = F.one_hot(allCodes, num_classes=256).float()
            # [2N * worldSize, 256]
            allLogits = torch.cat(allLogitsList)

            # [2N, 2N]
            loss = allLogits @ allCodes.T

            swapLoss.append(loss.diagonal(N).mean() + loss.diagonal(-N).mean())

            rejectLoss.append(-loss[mask].sum() / N)


            # [1] -> []
            interEntropy = Categorical(probs=torch.cat((leftSub, rightSub)).softmax(-1).mean(0, keepdim=True)).entropy().mean()
            regLoss.append(-interEntropy)
            resultDict.update({f"reg_{i}": -interEntropy, f"swap_{i}": swapLoss[-1], f"reject_{i}": rejectLoss[-1]})

        regLoss = sum(regLoss)
        swapLoss = sum(swapLoss)
        rejectLoss = sum(rejectLoss)

        return regLoss + swapLoss + rejectLoss, resultDict


    def forward(self, *, z: torch.Tensor, b: torch.Tensor, y: torch.Tensor, **_):
        left, right = self.splitFeatureInTwoViews(z, y)
        mapLoss, resultDict = self.mapping(left, right)

        trainLoss, resultDict1 = self.contrastive(left, right)

        resultDict.update(resultDict1)
        return mapLoss + trainLoss, resultDict
