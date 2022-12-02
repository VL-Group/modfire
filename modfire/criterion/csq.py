from scipy.linalg import hadamard
import torch
from torch import nn
import torch.nn.functional as F

from .utils import pairwiseHamming, CriterionRegistry


@CriterionRegistry.register
class CSQ(nn.Module):
    centroids: torch.Tensor
    def __init__(self, bits: int, numClasses: int, _lambda: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("centroids", self.generateCentroids(bits, numClasses))
        self._lambda = _lambda

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # [N, C]
        centerLoss = F.binary_cross_entropy_with_logits(x, y.float() @ self.centroids)
        quantizationError = F.mse_loss(x.tanh(), x.sign())
        return centerLoss + quantizationError, { "centerLoss": centerLoss, "qError": quantizationError }

    @staticmethod
    def generateCentroids(bits: int, numClasses: int):
        if numClasses > 2 * bits:
            return CSQ._randomCode(bits, numClasses).float()
        else:
            return CSQ._hadamardCode(bits, numClasses).float()

    @staticmethod
    def _randomCode(bits: int, numClasses: int) -> torch.Tensor:
        best = None
        bestDis = float("-inf")
        for _ in range(100):
            sampled = torch.rand((numClasses, bits)) > 0.5
            distance = pairwiseHamming(sampled)
            if float(distance[distance > 0].min()) > bestDis:
                best = sampled
                bestDis = float(distance[distance > 0].min())
        if best is None:
            raise RuntimeError("Failed to create random centers.")
        return best

    @staticmethod
    def _hadamardCode(bits: int, numClasses: int) -> torch.Tensor:
        # [bits, bits]
        H = hadamard(bits)
        H = torch.tensor(H)
        # [2bits, bits]
        H = torch.cat([H, -H])
        H = H[:numClasses]
        return H > 0

@CriterionRegistry.register
class CSQ_D(CSQ):
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
        def __init__(self, bits) -> None:
            super().__init__()
            self.m = bits // 8
            ffnNet = lambda: nn.Sequential(
                nn.Linear(8, 256),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 256)
            )
            self._net = nn.ModuleList(ffnNet() for _ in range(bits // 8))
            self._bitFlip = CSQ_D._randomBitFlip(bits, int(bits // 32) ** 2)

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

    def __init__(self, bits: int, numClasses: int, _lambda: float = 0.0001) -> None:
        super().__init__(bits, numClasses, _lambda)
        self.bitFlip = self._randomBitFlip(bits, int(bits // 32) ** 2)
        self.mapper = self._mapNet(bits)
        self.register_buffer("multiplier", (2 ** torch.arange(8)).long())
        self.register_buffer("permIdx", torch.randperm(bits))
        self.m = bits // 8
        self._ticker = 0

    @property
    def BitFlip(self) -> int:
        return self.mapper._bitFlip.BitFlip

    @BitFlip.setter
    def BitFlip(self, numBitsToFlip: int):
        self.mapper._bitFlip.BitFlip = numBitsToFlip
        self.bitFlip.BitFlip = numBitsToFlip

    def resetPermIdx(self):
        # reset permIdx
        self.permIdx.data.copy_(torch.randperm(self.m * 8, device=self.permIdx.device))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        self._ticker += 1
        if self._ticker % int(self.m * 16) == 0:
            self.resetPermIdx()
        # X are permuted on last dim according to permIdx
        x = x[:, self.permIdx]

        originalX = x.clone().detach()
        # [N, M * 256]
        decimalHat = self.mapper(x.detach(), True)
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

        hashCenter = self.centroids[:, self.permIdx].clone().detach()

        netLoss = list()
        decimalHatGrad = self.mapper(x, False)
        # M * [N, 256]
        splitted = torch.chunk(decimalHatGrad, self.m, -1)

        # 8 * [Class, D // 8]
        hashCenterSplitted = torch.chunk(hashCenter > 0, self.m, -1)

        numClasses = len(self.centroids)

        for subX, targetCenter in zip(splitted, hashCenterSplitted):
            # bits to decimal
            # [class]
            targetCenter = (targetCenter * self.multiplier).sum(-1)
            # calculate ce from x to all centers
            # [N, 256, class]
            xToAllCenter = subX[..., None].expand(*subX.shape, numClasses)

            # [N, class]
            centerForAllX = targetCenter.expand(len(subX), numClasses)
            # [N, class]
            loss = F.cross_entropy(xToAllCenter, centerForAllX, reduction="none")
            mask = (y > 0)
            floatMask = mask.float()
            # [N, class]
            meanWeight = floatMask / floatMask.sum(-1, keepdim=True)
            # mask loss that label is zero
            # weighted sum over each sample for positive class
            # then mean over whole batch
            loss = ((loss * mask) * meanWeight).sum() / len(loss)
            netLoss.append(loss)

        codesToCenterDistance = list()
        for i in range(len(self.centroids)):
            thisClassFeatures = x[y[:, i] > 0]
            featureToCenterHamming = ((thisClassFeatures > 0) != (hashCenter[i] > 0)).sum(-1)
            # [?, D] with [D] -> sum() -> [?]
            codesToCenterDistance.append(featureToCenterHamming)
        # [?]
        codesToCenterDistance = torch.cat(codesToCenterDistance).float().mean()

        netLoss = sum(netLoss)
        mapLoss = sum(mapLoss)

        return netLoss + mapLoss, { "netLoss": netLoss, "mapLoss": mapLoss, "hitRate": sum(hitRate) / self.m, "codesToCenterDistance": codesToCenterDistance }
