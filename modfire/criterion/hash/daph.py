import logging
from typing import Any
from itertools import product
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import IterDataPipe

from modfire import Consts
import modfire.train.hooks
from modfire.model import ModelRegistry

from ..utils import pairwiseHamming, CriterionRegistry

logger = logging.getLogger(Consts.Name)


@CriterionRegistry.register
class DAPH(nn.Module, modfire.train.hooks.EpochStartHook, modfire.train.hooks.EpochFinishHook, modfire.train.hooks.StepStartHook, modfire.train.hooks.StepFinishHook):
    """
        Fumin Shen, Xin Gao, Li Liu, Yang Yang, Heng Tao Shen: Deep Asymmetric Pairwise Hashing. ACM Multimedia 2017: 1522-1530
    """
    U: torch.Tensor
    Z: torch.Tensor
    Y: torch.Tensor
    I: torch.Tensor
    B: torch.Tensor
    H: torch.Tensor
    def __init__(self, bits: int, alpha: float, beta: float, gamma: float, _lambda: float, modelKey, modelParams) -> None:
        super().__init__()
        self.bits = bits
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._lambda = _lambda
        self.topModel = ModelRegistry.get(modelKey)(**modelParams)

    def beforeRun(self, step: int, epoch: int, trainer, *_, trainSet, logger):
        self.register_buffer("U", torch.zeros(len(trainSet), self.bits))
        self.register_buffer("Z", torch.zeros(len(trainSet), self.bits))
        self.register_buffer("Y", torch.zeros(len(trainSet), self.bits))
        self.register_buffer("I", torch.eye(self.bits))
        self.register_buffer("B", torch.randn(len(trainSet), self.bits).sign())
        self.register_buffer("H", torch.randn(len(trainSet), self.bits).sign())

    def calcLoss(self):
        s = (self.Y @ self.Y.t() > 0).float()
        inner_product = self.U @ self.Z.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        quantization_loss = (self.U - self.B).pow(2) + (self.Z - self.H).pow(2)
        quantization_loss = self.alpha * quantization_loss.mean()

        regularization_loss = self.gamma * (self.B - self.H).pow(2).mean()

        independence_loss = (self.U.t() @ self.U / self.U.shape[0] - self.I).pow(2) + \
                            (self.Z.t() @ self.Z / self.Z.shape[0] - self.I).pow(2)
        independence_loss = self._lambda * independence_loss.mean()

        balance_loss = self.U.sum(dim=0).pow(2) + self.Z.sum(dim=0).pow(2)
        balance_loss = self.beta * balance_loss.mean()

        return likelihood_loss + quantization_loss + regularization_loss + independence_loss + balance_loss

    def epochStart(self, step: int, epoch: int, trainer, *args: Any, logger, **kwds: Any) -> Any:
        if epoch % 2 == 0:
            self.topModel.train()
            trainer._model.eval()
        else:
            self.topModel.eval()
            trainer._model.train()


    def stepStart(self, step: int, epoch: int, trainer, *_, logger, inputs, **kwds: Any) -> Any:
        targets, idx, images = inputs
        trainer.zero_grad()
        u = self.topModel(images.to(trainer.rank, non_blocking=True))

        self._tempU = u



    def forward(self, x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor):
        loss = (x - self.H[idx]).pow(2).mean()
        return loss
