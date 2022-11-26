import torch
from torchdata.dataloader2 import DataLoader2
from torch.utils.data import IterDataPipe
from rich.progress import Progress
from vlutils.metrics.meter import Meters

from modfire.config import Config
from modfire.model.base import BaseWrapper
from modfire.dataset import Database

class Validator:
    def __init__(self, config: Config):
        self.config = config
        self.numReturns = config.Train.NumReturns
        self._meter = Meters(handlers=[
            mAP(config.Train.NumReturns),
            Precision(config.Train.NumReturns),
            Recall(config.Train.NumReturns),
            Visualization()
        ])

    @torch.no_grad()
    def validate(self, model: BaseWrapper, database: Database, queries: QuerySet, progress: Progress):
        model.eval()
        self._meter.reset()

        model.add(database.DataPipe, progress)
        rankList = model.search(queries, self.numReturns, progress)
        truePositives = database.judge(queries.Info, rankList)
        self._meter(truePositives)
        return self._meter.results(), self._meter.summary()
