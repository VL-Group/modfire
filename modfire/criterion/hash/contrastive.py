import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from modfire import Consts

from ..utils import pairwiseHamming, CriterionRegistry, pairwiseCosine, pariwiseAffinity

logger = logging.getLogger(Consts.Root)


@CriterionRegistry.register
class Contrastive(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, *, b: torch.Tensor, y: torch.Tensor, **_):
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
        return loss.mean(), { }

    # def forward(self, *, b: torch.Tensor, y: torch.Tensor, **_):
    #     # [n, n, d]
    #     samePart = (b[:, None] > 0) == (b > 0)

    #     # let zi to be zj, if zi-zj is a positive pair.
    #     positiveTarget = (b > 0).clone().detach().expand_as(samePart)
    #     # let zi not to be zj, if zi-zj is a negative pair.
    #     negativeTarget = ((b > 0)[:, None].clone().detach().expand_as(samePart) * ~samePart) + ((torch.randn_like(b) > 0)[:, None].clone().detach().expand_as(samePart) * samePart)

    #     positiveLoss = F.binary_cross_entropy_with_logits(b[:, None].expand_as(samePart), positiveTarget.float(), reduction="none")
    #     negativeLoss = F.binary_cross_entropy_with_logits(b[:, None].expand_as(samePart), negativeTarget.float(), reduction="none")

    #     # label, equal pick positive, inequal pick negative
    #     # [N, N, 1]
    #     affinity = (y[..., None] == y)[..., None]
    #     # [N, N, 1]
    #     mask = ~torch.eye(len(b), dtype=torch.bool, device=b.device)[..., None]

    #     loss = ((positiveLoss * affinity) + (negativeLoss * ~affinity)) * mask

    #     return loss.mean(), { }


@CriterionRegistry.register
class Contrastive_D(nn.Module):
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
