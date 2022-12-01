from typing import Callable

import torch
from torch import nn
from vlutils.base import Registry


def pairwiseHamming(X: torch.Tensor):
    X = X.float() * 2 - 1
    # [N, N, D]
    pairwise = (X.shape[-1] - X @ X.T) / 2
    # mask diagonal
    pairwise[torch.eye(len(pairwise), dtype=torch.bool, device=X.device)] = -1
    return pairwise


class CriterionRegistry(Registry[Callable[..., nn.Module]]):
    pass
