from typing import Any, Tuple
import abc
import os

import torch
from torch.utils.data import IterDataPipe
from vlutils.saver import StrPath


class TrainSet(abc.ABC):
    @property
    @abc.abstractmethod
    def DataPipe(self) -> IterDataPipe:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def BatchSize(self) -> int:
        raise NotImplementedError

    def __len__(self):
        """If it is too big to determine dataset length, return -1
        """
        return -1


class QuerySet(abc.ABC):
    @property
    @abc.abstractmethod
    def DataPipe(self) -> IterDataPipe:
        """Return an IterDataPipe over whole query set.
            NOTE: The returned datapipe returns (i, img) where i is unique index and img is corresponding image in query set.

        Returns:
            IterDataPipe
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def BatchSize(self) -> int:
        raise NotImplementedError

    def __len__(self):
        """If it is too big to determine dataset length, return -1
        """
        return -1

    @abc.abstractmethod
    def info(self, indices) -> Any:
        """Return queries' info for Database.judge()
        Args:
            indices (torch.Tensor): [?] indices of queries, which could be obtained by DataPipe.

        Returns:
            Any: May be labels, texts, etc., should have the same order with self.DataPipe.
        """
        raise NotImplementedError


class Database(abc.ABC):
    @property
    @abc.abstractmethod
    def DataPipe(self) -> IterDataPipe:
        """Return an IterDataPipe over whole database.
            NOTE: The returned datapipe returns (i, img) where i is unique index and img is corresponding image in database.

        Returns:
            IterDataPipe
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def BatchSize(self) -> int:
        raise NotImplementedError

    def __len__(self):
        """If it is too big to determine dataset length, return -1
        """
        return -1

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

    def __init__(self, root: StrPath):
        super().__init__()
        self.root = root
        if not self.check():
            raise IOError(f"You have not organized `{self.__class__.__name__}` in the specified directory `{self.root}`.{os.linesep}"\
                f"If you want to prepare it, call `modfire dataset {self.__class__.__name__} --root=\"{self.root}\"`.")

    @property
    @abc.abstractmethod
    def TrainSet(self) -> TrainSet:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def QuerySet(self) -> QuerySet:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Database(self) -> Database:
        raise NotImplementedError
