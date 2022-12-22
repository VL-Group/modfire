import logging
from typing import Any

from scipy.linalg import hadamard
import torch
from torch import nn
import torch.nn.functional as F

from modfire import Consts
import modfire.train.hooks

from ..utils import pairwiseSquare, CriterionRegistry, pariwiseAffinity

logger = logging.getLogger(Consts.Root)

@CriterionRegistry.register
class DSH(nn.Module, modfire.train.hooks.BeforeRunHook):
    """
        Haomiao Liu, Ruiping Wang, Shiguang Shan, Xilin Chen: Deep Supervised Hashing for Fast Image Retrieval. CVPR 2016: 2064-2072
    """
    def __init__(self, bits: int, numClasses: int, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

        self.fc = nn.Linear(bits, numClasses, bias=False)

    def forward(self, *, z: torch.Tensor, b: torch.Tensor, y: torch.Tensor, **_):
        predict = self.fc(b)

        loss1 = F.cross_entropy(predict, y.argmax(-1))

        loss2 = ((1 - z.abs()) ** 3).abs().mean()

        return loss1 + self.alpha * loss2, {"loss1": loss1, "loss2": loss2}
