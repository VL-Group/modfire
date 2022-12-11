import logging
from typing import Any, Tuple, List, Callable, Union
import abc
import os
from enum import Enum
from contextlib import contextmanager

from vlutils.base import Registry
import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper
from vlutils.saver import StrPath
from torchvision.datasets import folder
from .utils import _defaultEvalDataPipe, _defaultTrainingDataPipe


class Split(Enum):
    Train = 0,
    Query = 1,
    Database = 2
    @property
    def TxtConst(self) -> str:
        _consts = {
            Split.Train: "train.txt",
            Split.Query: "query.txt",
            Split.Database: "database.txt",
        }
        return _consts[self]


class SplitBase(abc.ABC):
    def __init__(self, samples: Any, labels: Any, batchSize: int, loadImg: Callable[..., Any], pipeline: Callable[..., IterDataPipe]):
        super().__init__()
        self._samples = samples
        self._labels = labels
        self._batchSize = batchSize
        self._pipeline = pipeline
        self._loadImg = loadImg

    @property
    def DataPipe(self) -> IterDataPipe:
        return self._pipeline(IterableWrapper(self._labels).zip(IterableWrapper(self._samples)), self._loadImg, self._batchSize)

    @property
    def BatchSize(self) -> int:
        return self._batchSize

    def __len__(self):
        """If it is too big to determine dataset length, return -1
        """
        return len(self._samples)

    @contextmanager
    def device(self, device):
        originalDevice = self._labels.device
        self._labels = self._labels.to(device)
        yield
        self._labels = self._labels.to(originalDevice)


class TrainSplit(SplitBase, abc.ABC):
    @property
    @abc.abstractmethod
    def NumClass(self) -> int:
        return -1

class QuerySplit(SplitBase, abc.ABC):
    @property
    def DataPipe(self) -> IterDataPipe:
        return self._pipeline(IterableWrapper(self._samples).enumerate(), self._loadImg, self._batchSize)
    def info(self, indices) -> Any:
        """Return queries' info for Database.judge()
        Args:
            indices (torch.Tensor): [?] indices of queries, which could be obtained by DataPipe.

        Returns:
            Any: May be labels, texts, etc., should have the same order with self.DataPipe.
        """
        # [Nq, nClass]
        return self._labels[indices]


class Database(SplitBase, abc.ABC):
    @property
    def DataPipe(self) -> IterDataPipe:
        return self._pipeline(IterableWrapper(self._samples).enumerate(), self._loadImg, self._batchSize)
    def judge(self, queryInfo: Any, rankList: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return true positives based on query info.

        Args:
            queryInfo (Any): Information of query, may be indices, labels, etc.
            rankList (torch.Tensor): [Nq, numReturns] indices, each row represents a rank list of top K from database. Indices are obtained by DatPipe.

        Returns:
            torch.Tensor: [Nq, numReturns] true positives.
            torch.Tensor: [Nq], Number of all trues w.r.t. query.
        """
        # NOTE: Here, queryInfo is label of queries.
        # [Nq, k, nClass]
        databaseLabels = self._labels[rankList]
        # [Nq, k]
        matching = torch.einsum("qc,qkc->qk", queryInfo, databaseLabels) > 0
        # [Nq, Nb] -> [Nq]
        numAllTrues = ((queryInfo @ self._labels.T) > 0).sum(-1)
        return matching, numAllTrues


class Dataset(abc.ABC):
    @abc.abstractmethod
    def check(self) -> bool:
        """Check whether the dataset is downloaded and verified.

        Returns:
            bool
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def prepare(root: StrPath, logger=logging) -> bool:
        """Check whether the dataset is downloaded and verified.

        Returns:
            bool
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Semantics(self) -> List[str]:
        raise NotImplementedError


    def _loadImg(self, inputs):
        i, img = inputs
        return i, folder.default_loader(img)

    def __init__(self, root: StrPath, mode: str, batchSize: int, pipeline = None):
        super().__init__()
        self.root = root
        self.mode = Split[mode]
        self.batchSize = batchSize
        self.pipeline = pipeline or (_defaultTrainingDataPipe if self.mode == Split.Train else _defaultEvalDataPipe)
        if not self.check():
            raise IOError(f"You have not organized `{self.__class__.__name__}` in the specified directory `{self.root}`.{os.linesep}"\
                f"If you want to prepare it, call `modfire dataset {self.__class__.__name__} --root=\"{self.root}\"`.")

    def __repr__(self) -> str:
        return "<%s, root=%s, mode=%s, batchSize=%s, pipeline=%s>" % (self.__class__.__name__, self.root, self.mode, self.batchSize, self.pipeline)

    def __str__(self) -> str:
        return "<%s, root=%s, mode=%s, batchSize=%s, pipeline=%s>" % (self.__class__.__name__, self.root, self.mode, self.batchSize, self.pipeline)

    @property
    def Split(self) -> Union[TrainSplit, QuerySplit, Database]:
        if self.mode == Split.Train:
            return self.TrainSplit
        elif self.mode == Split.Query:
            return self.QuerySplit
        elif self.mode == Split.Database:
            return self.Database
        else:
            raise AttributeError(f"Mode not correct: {self.mode}.")

    @property
    @abc.abstractmethod
    def TrainSplit(self) -> TrainSplit:
        if self.mode != Split.Train:
            raise ValueError(f"You try to create a training split from a dataset with mode {self.mode}.")

    @property
    @abc.abstractmethod
    def QuerySplit(self) -> QuerySplit:
        if self.mode != Split.Query:
            raise ValueError(f"You try to create a query split from a dataset with mode {self.mode}.")

    @property
    @abc.abstractmethod
    def Database(self) -> Database:
        if self.mode != Split.Database:
            raise ValueError(f"You try to create a database split from a dataset with mode {self.mode}.")

class DatasetRegistry(Registry[Callable[..., Dataset]]):
    pass
