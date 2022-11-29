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
    def __init__(self, bits: int, numClasses: int) -> None:
        super().__init__()
        self.register_buffer("centroids", self.generateCentroids(bits, numClasses))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # [N, C]
        logits = x @ self.centroids.T
        return F.cross_entropy(logits, y)

    @staticmethod
    def generateCentroids(bits: int, numClasses: int):
        if numClasses > 2 * bits:
            return CSQ._randomCode(bits, numClasses).float() * 2 - 1
        else:
            return CSQ._hadamardCode(bits, numClasses).float() * 2 - 1

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
