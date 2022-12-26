import logging
from typing import Any

from scipy.linalg import hadamard
import torch
from torch import nn
import torch.nn.functional as F

from modfire import Consts
import modfire.train.hooks
from modfire.utils import ConcatTensor

from ..utils import pairwiseInnerProduct, CriterionRegistry, pariwiseAffinity

logger = logging.getLogger(Consts.Root)

@CriterionRegistry.register
class DSDH(nn.Module, modfire.train.hooks.EpochStartHook):
    """
        Qi Li, Zhenan Sun, Ran He, Tieniu Tan: Deep Supervised Discrete Hashing. NIPS 2017: 2482-2491
    """
    def __init__(self, alpha: float = 0.1, dccIter: int = 10):
        raise NotImplementedError
        super().__init__()
        self.alpha = alpha
        self.dccIter = dccIter
        self.B = ConcatTensor()
        self.Y = ConcatTensor()

    def epochStart(self, step: int, epoch: int, trainer, *args: Any, logger, **kwds: Any) -> Any:
        raise NotImplementedError
        # update band w
        B = self.B.Value
        for dit in range(self.dccIter):
            # W-step
            W = torch.inverse(B.T @ B + config["nu"] / config["mu"] * torch.eye(bit).to(device)) @ B @ self.Y.t()

            for i in range(B.shape[0]):
                P = W @ self.Y + config["eta"] / config["mu"] * self.U
                p = P[i, :]
                w = W[i, :]
                W_prime = torch.cat((W[:i, :], W[i + 1:, :]))
                B_prime = torch.cat((B[:i, :], B[i + 1:, :]))
                B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

        self.B = B
        self.W = W

    def forward(self, *, b: torch.Tensor, y: torch.Tensor, **_):
        raise NotImplementedError
        # [N, N]
        affinity = pariwiseAffinity(y).float()
        similarity = pairwiseInnerProduct(b) * 0.5

        likelihoodLoss = ((1 + (-(similarity.abs())).exp()).log() + similarity.clamp_min(0) - affinity * similarity).mean()

        clLoss =
        quantizationLoss = ((b - b.sign()) ** 2).mean()

        return likelihoodLoss + self.alpha * quantizationLoss, {"likelihoodLoss": likelihoodLoss, "quantizationLoss": quantizationLoss}
