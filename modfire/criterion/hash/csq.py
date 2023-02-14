import logging
from typing import Any

from scipy.linalg import hadamard
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseHamming, CriterionRegistry

logger = logging.getLogger(Consts.Root)

@CriterionRegistry.register
class CSQ(nn.Module, modfire.train.hooks.BeforeRunHook):
    """
        Li Yuan, Tao Wang, Xiaopeng Zhang, Francis E. H. Tay, Zequn Jie, Wei Liu, Jiashi Feng: Central Similarity Quantization for Efficient Image and Video Retrieval. CVPR 2020: 3080-3089.
    """
    centroids: torch.Tensor
    def __init__(self, bits: int, numClasses: int, _lambda: float = 1e-4):
        super().__init__()
        self.bits = bits
        self.register_buffer("centroids", self.generateCentroids(bits, numClasses))
        self._lambda = _lambda

    def beforeRun(self, *_, trainSet, logger, **__) -> Any:
        logger.debug("Call `CSQ.beforeRun()`.")
        if trainSet.NumClass != len(self.centroids):
            raise ValueError("The dataset's `NumClass` not equals to centriods' number.")


    def meanOfCode(self, y: torch.Tensor):
        # [n, bits]
        return ((y @ self.centroids) > (y.sum(-1, keepdim=True) / 2)).float()

    def forward(self, *, z: torch.Tensor, b: torch.Tensor, y: torch.Tensor, **_):
        centerLoss = F.binary_cross_entropy_with_logits(z, self.meanOfCode(y))
        quantizationError = F.mse_loss(b, b.sign())
        return centerLoss + quantizationError, { "centerLoss": centerLoss, "qError": quantizationError }

    @staticmethod
    def generateCentroids(bits: int, numClasses: int):
        if numClasses > 2 * bits:
            logger.debug("Use random center.")
            return CSQ._randomCode(bits, numClasses).float()
        else:
            logger.debug("Use Hadamard center.")
            return CSQ._hadamardCode(bits, numClasses).float()

    @staticmethod
    def _randomCode(bits: int, numClasses: int) -> torch.Tensor:
        best = None
        bestDis = float("-inf")
        for _ in range(10000):
            sampled = torch.randn((numClasses, bits)) > 0
            distance = pairwiseHamming(sampled)
            if float(distance[distance > 0].min()) > bestDis:
                best = sampled
                bestDis = float(distance[distance > 0].min())
        if best is None:
            raise RuntimeError("Failed to create random centers.")
        logger.debug("Random center min distance: %.2f", bestDis)
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
class CSQ_D(CSQ, modfire.train.hooks.EpochFinishHook):
    """
        Xiaosu Zhu, Jingkuan Song, Yu Lei, Lianli Gao, Heng Tao Shen: A Lower Bound of Hash Codes' Performance. NeurIPS 2022.
    """
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

    def __init__(self, bits: int, numClasses: int, addRegularization: bool, _lambda: float = 0.0001):
        super().__init__(bits, numClasses, _lambda)
        self.bitFlip = self._randomBitFlip(bits, int(bits // 32) ** 2)
        self.mapper = self._mapNet(bits)
        self.register_buffer("multiplier", (2 ** torch.arange(8)).long())
        self.register_buffer("permIdx", torch.randperm(bits))
        self.m = bits // 8
        self.addRegularization = addRegularization

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

    def forward(self, *, b: torch.Tensor, y: torch.Tensor, **_):
        # X are permuted on last dim according to permIdx
        z = b[:, self.permIdx]

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

        hashCenter = self.centroids[:, self.permIdx].clone().detach()

        decimalHatGrad = self.mapper(z, False)
        # M * [N, 256]
        splitted = torch.chunk(decimalHatGrad, self.m, -1)

        # 8 * [Class, D // 8]
        hashCenterSplitted = torch.chunk(hashCenter > 0, self.m, -1)

        numClasses = len(self.centroids)

        netLoss = list()
        regLoss = list()

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
            # mask loss that label is zero
            # weighted sum over each sample for positive class
            # then mean over whole batch
            loss = (((loss * y)).sum(-1) / y.sum(-1)).sum() / len(loss)
            netLoss.append(loss)

            if self.addRegularization:
                # naive
                # meanLogits = subX.mean(0, keepdim=True)
                # entropy = Categorical(logits=meanLogits).entropy().mean()
                # regLoss.append(-entropy)

                # use probs
                meanProbs = subX.softmax(-1).mean(0, keepdim=True)
                entropy = Categorical(probs=meanProbs).entropy().mean()
                regLoss.append(-entropy)

                # use STE, like gumbel-softmax
                # see https://aclanthology.org/2022.acl-short.20.pdf
                # MLE estimator
                # [N, K]
                # oneHot = F.one_hot(subX.argmax(-1), num_classes=256).float()
                # ste = (oneHot - subX).detach() + subX
                # meanProbs = ste.mean(0, keepdim=True)
                # entropy = Categorical(probs=meanProbs).entropy().mean()
                # regLoss.append(-1e-8 * entropy)

                # jackknife estimator
                # oneHot = F.one_hot(subX.argmax(-1), num_classes=256).float()
                # ste = (oneHot - subX).detach() + subX
                # meanProbs = ste.mean(0, keepdim=True)
                # entropy = Categorical(probs=meanProbs).entropy().sum()
                # partialEntropys = list()
                # for i in range(len(subX)):
                #     mask = torch.ones([len(subX)], dtype=torch.bool)
                #     mask[i] = False
                #     partial = ste[mask]
                #     partialProbs = partial.mean(0, keepdim=True)
                #     partialEntropy = Categorical(probs=partialProbs).entropy().sum()
                #     partialEntropys.append(partialEntropy)
                # jackknife = len(subX) * entropy - (len(subX) - 1) / len(subX) * sum(partialEntropys)
                # regLoss.append(-1e-8 * jackknife)

                # meanProbs = subX.softmax(-1).mean(0, keepdim=True)
                # entropy = Categorical(probs=meanProbs).entropy().sum()
                # partialEntropys = list()
                # for i in range(len(subX)):
                #     mask = torch.ones([len(subX)], dtype=torch.bool)
                #     mask[i] = False
                #     partial = subX.softmax(-1)[mask]
                #     partialProbs = partial.mean(0, keepdim=True)
                #     partialEntropy = Categorical(probs=partialProbs).entropy().sum()
                #     partialEntropys.append(partialEntropy)
                # jackknife = len(subX) * entropy - (len(subX) - 1) / len(subX) * sum(partialEntropys)
                # regLoss.append(-jackknife)

        codesToCenterDistance = list()
        for i in range(len(self.centroids)):
            thisClassFeatures = z[y[:, i] > 0]
            featureToCenterHamming = ((thisClassFeatures > 0) != (hashCenter[i] > 0)).sum(-1)
            # [?, D] with [D] -> sum() -> [?]
            codesToCenterDistance.append(featureToCenterHamming)
        # [?]
        codesToCenterDistance = torch.cat(codesToCenterDistance).float().mean()

        netLoss = sum(netLoss)
        mapLoss = sum(mapLoss)
        regLoss = sum(regLoss)

        return netLoss + mapLoss + regLoss, { "netLoss": netLoss, "mapLoss": mapLoss, "regLoss": regLoss, "hitRate": sum(hitRate) / self.m, "codesToCenterDistance": codesToCenterDistance }
