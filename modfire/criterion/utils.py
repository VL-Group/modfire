from typing import Callable, Optional

import torch
from torch import nn
from vlutils.base import Registry


def pairwiseHamming(x: torch.Tensor, optionalY: Optional[torch.Tensor] = None):
    x = x.float() * 2 - 1
    if optionalY is None:
        # [N, N]
        pairwise = (x.shape[-1] - x @ x.T) / 2
        # mask diagonal
        pairwise[torch.eye(len(pairwise), dtype=torch.bool, device=x.device)] = -1
        return pairwise
    # [N1, N2]
    pairwise = (x.shape[-1] - x @ optionalY.T) / 2
    return pairwise

def pariwiseAffinity(y: torch.Tensor, optionalAnother: Optional[torch.Tensor] = None):
    if optionalAnother is None:
        # [N, N]
        affinity = (y @ y.T) > 0
        # mask diagonal
        affinity[torch.eye(len(affinity), dtype=torch.bool, device=affinity.device)] = False
        return affinity
    # [N1, N2]
    return (y @ optionalAnother.T) > 0

def pairwiseInnerProduct(x: torch.Tensor, optionalY: Optional[torch.Tensor] = None):
    if optionalY is None:
        # [N, N]
        pairwise = x @ x.T
        # mask diagonal
        pairwise[torch.eye(len(pairwise), dtype=torch.bool, device=pairwise.device)] = -1
        return pairwise
    # [N1, N2]
    return x @ optionalY.T

class CriterionRegistry(Registry[Callable[..., nn.Module]]):
    pass
