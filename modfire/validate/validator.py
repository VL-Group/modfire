import torch
from rich.progress import Progress
from vlutils.metrics.meter import Meters

from modfire.config import Config
from modfire.model.base import BaseWrapper
from modfire.dataset import Database, QuerySplit

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
    def validate(self, model: BaseWrapper, database: Database, queries: QuerySplit, progress: Progress):
        device = next(model.parameters()).device
        model.eval()
        self._meter.reset()
        model.reset()

        with database.device(device), queries.device(device):
            model.add(database, progress)
            for queryIndices, rankList in model.search(queries, self.numReturns, progress):
                truePositives, numAllTrues = database.judge(queries.info(queryIndices), rankList.to(device))
                if torch.any(numAllTrues < 1):
                    raise RuntimeError("We find a query is not relevant with any samples in database.")
                # remove outliers
                self._meter(truePositives, numAllTrues)

        model.train()
        return self._meter.results(), self._meter.summary()
