from typing import Callable

from scipy.linalg import hadamard
import torch
from torch import nn
import torch.nn.functional as F
from vlutils.base import Registry

from .utils import pairwiseHamming


class CriterionRegistry(Registry[Callable[..., nn.Module]]):
    pass


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
        return centerLoss + quantizationError

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
