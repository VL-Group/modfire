from typing import List

import torch
from vlutils.metrics.meter import Handler
from torchvision.io.image import write_png

__all__ = [
    "mAP",
    "Precision",
    "Recall",
    "Visualization"
]

class RankListBasedMetric(Handler):
    def __init__(self, numReturns: int, format: str = r"%.2f"):
        super().__init__(format)
        self._numReturns = numReturns
    def __repr__(self) -> str:
        return self.MetricInfo + ": " + str(self)

    @property
    def MetricInfo(self):
        raise NotImplementedError

    def check(self, truePositives: torch.Tensor):
        if self._numReturns > truePositives.shape[-1]:
            raise ValueError(f"Returned rank list has fewer samples than numReturns. (expected: {self._numReturns}, actual: {truePositives.shape[-1]})")
        elif self._numReturns < 0:
            return truePositives.float()
        return truePositives[:, :self._numReturns].float()


class mAP(RankListBasedMetric):
    @property
    def MetricInfo(self):
        if self._numReturns < 0:
            numReturns = "all"
        else:
            numReturns = str(self._numReturns)
        return f"mAP@{numReturns}"
    def handle(self, truePositives: torch.Tensor, *_, **__) -> List[float]:
        truePositives = self.check(truePositives)
        # [K]
        base = torch.arange(truePositives.shape[-1], device=truePositives.device) + 1
        # [N, K]
        accumulated = truePositives.cumsum(-1)
        # [N]
        ap = (accumulated / base * truePositives).sum(-1) / truePositives.sum(-1).clamp_min_(1.0)
        return ap.tolist()


class Precision(RankListBasedMetric):
    @property
    def MetricInfo(self):
        if self._numReturns < 0:
            numReturns = "all"
        else:
            numReturns = str(self._numReturns)
        return f"P@{numReturns}"
    def handle(self, truePositives: torch.Tensor, *_, **__) -> List[float]:
        truePositives = self.check(truePositives)
        return (truePositives.sum(-1) / truePositives.shape[-1]).tolist()


class Recall(RankListBasedMetric):
    @property
    def MetricInfo(self):
        if self._numReturns < 0:
            numReturns = "all"
        else:
            numReturns = str(self._numReturns)
        return f"R@{numReturns}"
    def handle(self, truePositives: torch.Tensor, numAllTrues: torch.Tensor) -> List[float]:
        truePositives = self.check(truePositives)
        return (truePositives.sum(-1) / numAllTrues).tolist()


class Visualization(Handler):
    pass
