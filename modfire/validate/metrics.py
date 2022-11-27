from typing import List

import torch
from vlutils.metrics.meter import Handler


class RankListBasedMetric(Handler):
    def __init__(self, numReturns: int, format: str = r"%.2f"):
        super().__init__(format)
        self._numReturns = numReturns
    def check(self, truePositives: torch.Tensor):
        if self._numReturns > truePositives.shape[-1]:
            raise ValueError(f"Returned rank list has fewer samples than numReturns. (expected: {self._numReturns}, actual: {truePositives.shape[-1]})")
        return truePositives[:, :self._numReturns].float()


class mAP(RankListBasedMetric):
    def handle(self, truePositives: torch.Tensor, *_, **__) -> List[float]:
        truePositives = self.check(truePositives)
        # [K]
        base = torch.arange(truePositives.shape[-1], device=truePositives.device) + 1
        # [N, K]
        accumulated = truePositives.cumsum(-1)
        # [N]
        ap = (base / accumulated).sum(-1) / truePositives.sum(-1).clamp_min_(1.0)
        return ap.tolist()


class Precision(RankListBasedMetric):
    def handle(self, truePositives: torch.Tensor, *_, **__) -> List[float]:
        truePositives = self.check(truePositives)
        return (truePositives.sum(-1) / truePositives.shape[-1]).tolist()


class Recall(RankListBasedMetric):
    def handle(self, truePositives: torch.Tensor, numAllTrues: torch.Tensor) -> List[float]:
        truePositives = self.check(truePositives)
        return (truePositives.sum(-1) / numAllTrues).tolist()
