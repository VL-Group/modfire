import torch
from rich.progress import Progress
from vlutils.metrics.meter import Meters

from modfire.config import Config
from modfire.model.base import BaseWrapper
from modfire.dataset import Database, QuerySet

from .metrics import mAP, Precision, Recall, Visualization

class Validator:
    def __init__(self, numReturns: int):
        self.numReturns = numReturns
        self._meter = Meters(handlers=[
            mAP(numReturns),
            Precision(numReturns),
            Recall(numReturns),
            # Visualization()
        ])

    @torch.no_grad()
    def validate(self, model: BaseWrapper, database: Database, queries: QuerySet, progress: Progress):
        model.eval()
        self._meter.reset()
        model.reset()

        model.add(database, progress)
        queryIndices, rankList = model.search(queries, self.numReturns, progress)
        truePositives, numAllTrues = database.judge(queries.info(queryIndices), rankList)
        self._meter(truePositives, numAllTrues)
        return self._meter.results(), self._meter.summary()
