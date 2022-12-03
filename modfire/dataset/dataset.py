from typing import Any, Tuple, List
import abc
import os
from enum import Enum

import torch
from torch.utils.data import IterDataPipe
from vlutils.saver import StrPath

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
    _batchSize: int
    @property
    @abc.abstractmethod
    def DataPipe(self) -> IterDataPipe:
        raise NotImplementedError

    @property
    def BatchSize(self) -> int:
        return self._batchSize

    def __len__(self):
        """If it is too big to determine dataset length, return -1
        """
        return -1


class TrainSplit(SplitBase, abc.ABC):
    pass

class QuerySplit(SplitBase, abc.ABC):
    @abc.abstractmethod
    def info(self, indices) -> Any:
        """Return queries' info for Database.judge()
        Args:
            indices (torch.Tensor): [?] indices of queries, which could be obtained by DataPipe.

        Returns:
            Any: May be labels, texts, etc., should have the same order with self.DataPipe.
        """
        raise NotImplementedError


class Database(SplitBase, abc.ABC):
    @abc.abstractmethod
    def judge(self, queryInfo: Any, rankList: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return rank list matching result

        Args:
            queryInfo (Any): Information of query, may be indices, labels, etc.
            rankList (torch.Tensor): [Nq, numReturns] indices, each row represents a rank list of top K from database. Indices are obtained by DatPipe.

            Returns:
                torch.Tensor: [Nq, numReturns] true positives.
                torch.Tensor: [Nq], Number of all trues w.r.t. query.
        """
        raise NotImplementedError


class Dataset(abc.ABC):
    @abc.abstractmethod
    def check(self) -> bool:
        """Check whether the dataset is downloaded and verified.

        Returns:
            bool
        """

    @abc.abstractmethod
    def prepare(self, root: StrPath):
        """Check whether the dataset is downloaded and verified.

        Returns:
            bool
        """

    @property
    @abc.abstractmethod
    def Semantics(self) -> List[str]:
        raise NotImplementedError

    def __init__(self, root: StrPath, mode: str, batchSize: int):
        super().__init__()
        self.root = root
        self.mode = Split[mode]
        self.batchSize = batchSize
        if not self.check():
            raise IOError(f"You have not organized `{self.__class__.__name__}` in the specified directory `{self.root}`.{os.linesep}"\
                f"If you want to prepare it, call `modfire dataset {self.__class__.__name__} --root=\"{self.root}\"`.")

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
