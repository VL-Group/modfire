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

def pairwiseSquare(x: torch.Tensor, optionalY: Optional[torch.Tensor] = None):
    if optionalY is None:
        # [N, N]
        pairwise = x @ x.T
        l2 = (x ** 2).sum(-1)
        distance = l2[..., None] - 2 * pairwise + l2
        # mask diagonal
        distance[torch.eye(len(pairwise), dtype=torch.bool, device=pairwise.device)] = -1
        return distance
    # [N1, N2]
    pairwise = x @ optionalY.T
    x2 = (x ** 2).sum(-1, keepdim=True)
    y2 = (optionalY ** 2).sum(-1)
    distance = x2 - 2 * pairwise + y2
    return distance

def pairwiseCosine(x: torch.Tensor, optionalY: Optional[torch.Tensor] = None):
    if optionalY is None:
        # [N, N]
        pairwise = x @ x.T
        l2 = (x ** 2).sum(-1)
        cosine = pairwise / (l2[..., None] * l2).sqrt()
        # mask diagonal
        cosine[torch.eye(len(pairwise), dtype=torch.bool, device=pairwise.device)] = -1
        return cosine
    # [N1, N2]
    pairwise = x @ optionalY.T
    x2 = (x ** 2).sum(-1, keepdim=True)
    y2 = (optionalY ** 2).sum(-1)
    cosine = pairwise / (x2 * y2).sqrt()
    return cosine

class CriterionRegistry(Registry[Callable[..., nn.Module]]):
    pass
