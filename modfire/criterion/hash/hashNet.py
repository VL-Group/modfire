import logging

import torch
from torch import nn

from modfire import Consts

from ..utils import CriterionRegistry

logger = logging.getLogger(Consts.Root)


@CriterionRegistry.register
class HashNet(nn.Module):
    """
        Zhangjie Cao, Mingsheng Long, Jianmin Wang, Philip S. Yu: HashNet: Deep Learning to Hash by Continuation. ICCV 2017: 5609-5618
    """
    def __init__(self, bits: int, alpha: float = 0.1):
        super().__init__()
        self.bits = bits
        self.alpha = alpha

    def forward(self, *, z: torch.Tensor, y: torch.Tensor, **_):
        u = torch.tanh(self.scale * z)
        S = (y @ y.t() > 0).float()
        dot_product = self.alpha * u @ u.t()
        mask_positive = S > 0
        mask_negative = (1 - S).bool()

        neg_log_probe = dot_product + torch.log(1 + torch.exp(-dot_product)) -  S * dot_product
        S1 = torch.sum(mask_positive.float())
        S0 = torch.sum(mask_negative.float())
        S = S0 + S1

        neg_log_probe[mask_positive] = neg_log_probe[mask_positive] * S / S1
        neg_log_probe[mask_negative] = neg_log_probe[mask_negative] * S / S0

        loss = torch.sum(neg_log_probe) / S
        return loss
