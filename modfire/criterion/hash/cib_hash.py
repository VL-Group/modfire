import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli, kl_divergence

from modfire import Consts

from ..utils import pairwiseHamming, CriterionRegistry, pairwiseCosine, pariwiseAffinity

logger = logging.getLogger(Consts.Root)


@CriterionRegistry.register
class CIBHash(nn.Module):
    def __init__(self, gamma: float, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma

    def jsLoss(self, left, right):
        left = Bernoulli(logits=left)
        right = Bernoulli(logits=right)
        return ((kl_divergence(left, right) + kl_divergence(right, left)) / 2).mean()

    def splitFeatureInTwoViews(self, z: torch.Tensor, y: torch.Tensor):
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
        batchSize = len(left)
        N = 2 * batchSize
        z = torch.cat((left, right))
        sim = F.cosine_similarity(z[:, None], z[None, :], -1) / self.temperature

        simIJ = torch.diag(sim, batchSize)
        simJI = torch.diag(sim, -batchSize)

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
class CIBHash_D(nn.Module):
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
            self._bitFlip = Contrastive_D._randomBitFlip(bits, int(bits // 32) ** 2)

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

    def __init__(self, bits: int, _lambda: float, temperature: float):
        super().__init__()
        self.bits = bits
        self._lambda = _lambda
        self.bitFlip = self._randomBitFlip(bits, int(bits // 32) ** 2)
        self.mapper = self._mapNet(bits)
        self.register_buffer("multiplier", (2 ** torch.arange(8)).long())
        self.register_buffer("permIdx", torch.randperm(bits))
        self.m = bits // 8
        self.temperature = temperature

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
        logger.debug("Call `CSQ_D.epochFinish()`.")
        if epoch:
            logger.debug("Reset permutation index in `CSQ_D`.")
            self.reset()

    def mappingAndEntropy(self, z: torch.Tensor, y: torch.Tensor):
        # X are permuted on last dim according to permIdx
        z = z[:, self.permIdx]

        originalX = z.clone().detach()
        # [N, M * 256]
        decimalHat = self.mapper(z.detach(), True)
        # M * [N, 256]
        splitted = torch.chunk(decimalHat, self.m, -1)
        rawSplit = torch.chunk(self.bitFlip(originalX), self.m, -1)

        mapLoss = list()
        hitRate = list()

        for subX, rawSubX in zip(splitted, rawSplit):
            binary = rawSubX > 0
            target = (self.multiplier * binary).sum(-1)
            mapLoss.append(F.cross_entropy(subX, target))
            hitRate.append((subX.argmax(-1) == target).float().mean())

        netLoss = list()
        decimalHatGrad = self.mapper(z, False)
        # M * [N, 256]
        splitted = torch.chunk(decimalHatGrad, self.m, -1)

        resultDict = {}

        for i, subX in enumerate(splitted):
            intraEntropy = list()
            for label in torch.unique(y):
                thisLabelOutputs = subX[y == label]
                # [1, 256]
                logits = thisLabelOutputs.mean(0, keepdim=True)
                # [1] -> []
                intraEntropy.append(Categorical(logits=logits).entropy().sum())
            interEntropy = Categorical(logits=subX.mean(0, keepdim=True)).entropy().sum()
            netLoss.append(-interEntropy + self._lambda * sum(intraEntropy))
            resultDict.update({f"intra_{i}": sum(intraEntropy), f"inter_{i}": interEntropy})

        netLoss = sum(netLoss)
        mapLoss = sum(mapLoss)

        resultDict.update({ "netLoss": netLoss, "mapLoss": mapLoss, "hitRate": sum(hitRate) / self.m })

        return netLoss + mapLoss, resultDict


    def forward(self, *, b: torch.Tensor, z: torch.Tensor, y: torch.Tensor, **_):
        # [N, N]
        logits = (b @ b.T) / self.temperature
        # [N, N]
        affinity = (y[..., None] == y).float()
        # [N, N]
        mask = ~torch.eye(len(b), dtype=torch.bool, device=b.device)
        # [N, N-1]
        similarity = logits[mask].reshape(len(b), -1)
        # [N, N-1]
        affinity = affinity[mask].reshape(len(b), -1)
        loss = F.cross_entropy(similarity, affinity.argmax(-1))

        auxiliary, resultDict = self.mappingAndEntropy(z, y)
        resultDict.update({"baseLoss": loss.mean()})
        return loss.mean() + auxiliary, resultDict
