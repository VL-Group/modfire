from typing import Any
import abc

import torch
from torch.utils.data import IterDataPipe, MapDataPipe


class TrainSet(abc.ABC):
    @property
    @abc.abstractmethod
    def DataPipe(self) -> MapDataPipe:
        raise NotImplementedError


class QuerySet(abc.ABC):
    @property
    @abc.abstractmethod
    def DataPipe(self) -> IterDataPipe:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Info(self) -> Any:
        """Return queries' info for Database.judge()
        Returns:
            Any: May be indices, labels, etc., should have the same order with self.DataPipe.
        """
        raise NotImplementedError


class Database(abc.ABC):
    @property
    @abc.abstractmethod
    def DataPipe(self) -> IterDataPipe:
        raise NotImplementedError

    @abc.abstractmethod
    def judge(self, queryInfo: Any, rankList: torch.Tensor) -> torch.Tensor:
        """Return rank list matching result

        Args:
            queryInfo (Any): Information of query, may be indices, labels, etc.
            rankList (torch.Tensor): [len(queries), K] indices, each row represents a rank list of top K from database

        Returns:
            torch.Tensor: [len(queries), K], True or False.
        """
        raise NotImplementedError


class Dataset(abc.ABC):
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
