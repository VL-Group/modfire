import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli, kl_divergence

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseHamming, CriterionRegistry, pairwiseCosine, pariwiseAffinity

logger = logging.getLogger(Consts.Root)


@CriterionRegistry.register
class CIBHash(nn.Module):
    def __init__(self, gamma: float, temperature: float, bits: int):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma

    def jsLoss(self, left, right):
        left = Bernoulli(logits=left)
        right = Bernoulli(logits=right)
        return ((kl_divergence(left, right) + kl_divergence(right, left)) / 2).mean()

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

    def contrastiveLoss(self, left: torch.Tensor, right: torch.Tensor):
        left = left.sigmoid() - 0.5
        right = right.sigmoid() - 0.5

        left = (left.sign() - left).detach() + left
        right = (right.sign() - right).detach() + right


        batchSize = len(left)
        N = 2 * batchSize
        z = torch.cat((left, right))

        cosine = z @ z.T

        sim = cosine / self.temperature

        simIJ = sim.diagonal(batchSize)
        simJI = sim.diagonal(-batchSize)

        mask = (~torch.eye(N, dtype=torch.bool))

        upper, lower = mask.diagonal(batchSize), mask.diagonal(-batchSize)
        upper.copy_(False)
        lower.copy_(False)


        positives = torch.cat((simIJ, simJI)).reshape(N, 1)
        negatives = sim[mask].reshape(N, -1)
        logits = torch.cat((positives, negatives), -1)
        loss = F.cross_entropy(logits, torch.zeros(N, device=logits.device, dtype=torch.long))
        return loss

    def forward(self, *, z: torch.Tensor, b: torch.Tensor, y: torch.Tensor, **_):
        left, right = self.splitFeatureInTwoViews(z, y)
        jsLoss = self.jsLoss(left, right)
        contrastiveLoss = self.contrastiveLoss(left, right)
        loss = self.gamma * jsLoss + contrastiveLoss
        return loss, { "jsLoss": jsLoss, "contrastiveLoss": contrastiveLoss }

@CriterionRegistry.register
class CIBHash_D(CIBHash, modfire.train.hooks.EpochFinishHook):
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
            self._bitFlip = CIBHash_D._randomBitFlip(bits, int(bits // 16) ** 2)

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
        super().__init__(gamma, temperature, bits)
        self.bits = bits
        self.bitFlip = self._randomBitFlip(bits, int(bits // 16) ** 2)
        self.mapper = self._mapNet(bits)
        self.register_buffer("multiplier", (2 ** torch.arange(8)).long())
        self.register_buffer("permIdx", torch.randperm(bits))
        self.m = bits // 8

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

    def mappingAndEntropy(self, left: torch.Tensor, right: torch.Tensor, **_):

        mapLoss = list()
        hitRate = list()

        regLoss = list()
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
            mapLoss.append(F.cross_entropy(subX, target))
            hitRate.append((subX.argmax(-1) == target).float().mean())

        originalRight = right.clone().detach()
        rightDec = self.mapper(right.detach(), True)
        rightSplit = torch.chunk(rightDec, self.m, -1)
        rightRawSplit = torch.chunk(self.bitFlip(originalRight), self.m, -1)

        for subX, rawSubX in zip(rightSplit, rightRawSplit):
            binary = rawSubX > 0
            target = (self.multiplier * binary).sum(-1)
            mapLoss.append(F.cross_entropy(subX, target))
            hitRate.append((subX.argmax(-1) == target).float().mean())



        leftDecGrad = self.mapper(left, False)
        # M * [N, 256]
        leftSplitted = torch.chunk(leftDecGrad, self.m, -1)

        rightDecGrad = self.mapper(right, False)
        # M * [N, 256]
        rightSplitted = torch.chunk(rightDecGrad, self.m, -1)

        for i, (leftSub, rightSub) in enumerate(zip(leftSplitted, rightSplitted)):
            # [1, 256]
            leftDistribution = Categorical(logits=leftSub)
            rightDistribution = Categorical(logits=rightSub)
            # [1] -> []
            intraEntropy = ((kl_divergence(leftDistribution, rightDistribution) + kl_divergence(rightDistribution, leftDistribution)) / 2).mean()
            interEntropy = Categorical(probs=torch.cat((leftSub, rightSub)).softmax(-1).mean(0, keepdim=True)).entropy().mean()
            regLoss.append(-interEntropy + intraEntropy)
            resultDict.update({f"intra_{i}": intraEntropy, f"inter_{i}": interEntropy})

        regLoss = sum(regLoss)
        mapLoss = sum(mapLoss)

        resultDict.update({ "regLoss": regLoss, "mapLoss": mapLoss, "hitRate": sum(hitRate) / self.m })

        return mapLoss + 1e2 * regLoss, resultDict


    def forward(self, *, z: torch.Tensor, b: torch.Tensor, y: torch.Tensor, **_):
        left, right = self.splitFeatureInTwoViews(z, y)
        jsLoss = self.jsLoss(left, right)
        regLoss, resultDict = self.mappingAndEntropy(left, right)

        left, right = self.splitFeatureInTwoViews(b, y)
        contrastiveLoss = self.contrastiveLoss(left, right)
        loss = self.gamma * jsLoss + contrastiveLoss
        resultDict.update({ "jsLoss": jsLoss, "contrastiveLoss": contrastiveLoss })
        return loss + regLoss, resultDict
