import abc
from typing import Optional, Tuple, Callable
import enum
import math

from rich.progress import Progress
import torch
from torch import nn
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from vlutils.base import Registry

from modfire.dataset import Database, QuerySet
from .searcher import BinarySearcher, PQSearcher


class ModelRegistry(Registry[Callable[..., nn.Module]]):
    pass


class ModelType(enum.Enum):
    Hash = 1,
    ProductQuantization = 2
    def __str__(self):
        return self.name

class BaseWrapper(nn.Module, abc.ABC):
    _dummy: torch.Tensor
    def __init__(self):
        super().__init__()
        self.register_buffer("_dummy", torch.empty([1]), persistent=False)
    @property
    @abc.abstractmethod
    def Type(self) -> ModelType:
        raise NotImplementedError
    @abc.abstractmethod
    def encode(self, image: torch.Tensor):
        raise NotImplementedError
    @abc.abstractmethod
    def add(self, database: Database, progress: Optional[Progress] = None):
        raise NotImplementedError
    @abc.abstractmethod
    def remove(self, ids: torch.Tensor):
        raise NotImplementedError
    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError
    @abc.abstractmethod
    def search(self, queries: QuerySet, numReturns: int, progress: Optional[Progress] = None) -> torch.Tensor:
        raise NotImplementedError


class BinaryWrapper(BaseWrapper):
    _byteTemplate: torch.Tensor
    def __init__(self, bits: int):
        super().__init__()
        self.database = BinarySearcher(bits)
        self.register_buffer("_byteTemplate", torch.tensor([int(2 ** x) for x in range(8)]))

    def Type(self) -> ModelType:
        return ModelType.Hash

    def boolToByte(self, x: torch.Tensor) -> torch.Tensor:
        """Convert D-dim bool tensor to byte tensor along the last dimension.

        Args:
            x (torch.Tensor): [..., D], D-dim bool tensor.

        Returns:
            torch.Tensor: [..., D // 8], Converted byte tensor.
        """
        return (x.reshape(*x.shape[:-1], -1, 8) * self._byteTemplate).sum(-1).byte()

    @torch.no_grad()
    def add(self, database: Database, progress: Optional[Progress] = None):
        if progress is not None:
            task = progress.add_task(f"[ Index ]", total=len(database), progress=f" {0:.1f}k", suffix="")
        dataLoader = DataLoader2(database.DataPipe, reading_service=MultiProcessingReadingService(num_workers=8, pin_memory=True))
        allFeatures = list()
        allIdx = list()
        total = 0
        for idx, image in dataLoader:
            # [N, bits]
            h = self.encode(image.to(self._dummy.device, non_blocking=True))
            allFeatures.append(h.cpu())
            allIdx.append(idx.cpu())
            inrement = len(h)
            total += inrement
            if progress is not None:
                progress.update(task, advance=inrement, progress=f" {total/1000:.1f}k")
        # [N, D]
        allFeatures = torch.cat(allFeatures)
        allIdx = torch.cat(allIdx)
        if progress is not None:
            progress.remove_task(task)
        return self.database.add(allFeatures.numpy(), allIdx.numpy())

    @torch.no_grad()
    def remove(self, ids: torch.Tensor):
        return self.database.remove(ids.numpy())

    def reset(self):
        return self.database.reset()

    @torch.no_grad()
    def search(self, queries: QuerySet, numReturns: int, progress: Optional[Progress] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if progress is not None:
            task = progress.add_task(f"[ Query ]", total=len(queries), progress=f" {0:.1f}k", suffix="")
        dataLoader = DataLoader2(queries.DataPipe, reading_service=MultiProcessingReadingService(num_workers=8, pin_memory=True))
        allFeatures = list()
        allIdx = list()
        total = 0
        for idx, image in dataLoader:
            # [N, bits]
            h = self.encode(image.to(self._dummy.device, non_blocking=True))
            allFeatures.append(h.cpu())
            allIdx.append(idx.cpu())
            inrement = len(h)
            total += inrement
            if progress is not None:
                progress.update(task, advance=inrement, progress=f" {total/1000:.1f}k")
        # [N, D]
        allFeatures = torch.cat(allFeatures)
        allIdx = torch.cat(allIdx)
        if progress is not None:
            progress.remove_task(task)
        return allIdx, torch.from_numpy(self.database.search(allFeatures.numpy(), numReturns))


class PQWrapper(BaseWrapper):
    codebook: nn.Parameter
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        self.codebook = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(m, k, d // m)))
        self.database = PQSearcher(self.codebook.cpu().numpy())

    def Type(self) -> ModelType:
        return ModelType.ProductQuantization

    def updateCodebook(self):
        self.database.assignCodebook(self.codebook.cpu().numpy())

    @torch.no_grad()
    def add(self, database: Database, progress: Optional[Progress] = None):
        if progress is not None:
            task = progress.add_task(f"[ Index ]", total=len(database), progress=f" {0:.1f}k", suffix="")
        dataLoader = DataLoader2(database.DataPipe, reading_service=MultiProcessingReadingService(num_workers=8, pin_memory=True))
        allFeatures = list()
        allIdx = list()
        total = 0
        for idx, image in dataLoader:
            # [N, D]
            x = self.encode(image.to(self._dummy.device, non_blocking=True))
            allFeatures.append(x.cpu())
            allIdx.append(idx.cpu())
            inrement = len(x)
            total += inrement
            if progress is not None:
                progress.update(task, advance=inrement, progress=f" {total/1000:.1f}k")
        # [N, D]
        allFeatures = torch.cat(allFeatures)
        allIdx = torch.cat(allIdx)
        if progress is not None:
            progress.remove_task(task)
        return self.database.add(allFeatures.numpy(), allIdx.numpy())

    @torch.no_grad()
    def remove(self, ids: torch.Tensor):
        return self.database.remove(ids.numpy())

    def reset(self):
        self.updateCodebook()
        return self.database.reset()

    @torch.no_grad()
    def search(self, queries: QuerySet, numReturns: int, progress: Optional[Progress] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if progress is not None:
            task = progress.add_task(f"[ Query ]", total=len(queries), progress=f" {0:4d}", suffix="")
        dataLoader = DataLoader2(queries.DataPipe, reading_service=MultiProcessingReadingService(num_workers=8, pin_memory=True))
        allFeatures = list()
        allIdx = list()
        total = 0
        for idx, image in dataLoader:
            # [D]
            x = self.encode(image.to(self._dummy.device, non_blocking=True))
            allFeatures.append(x.cpu())
            allIdx.append(idx.cpu())
            inrement = len(x)
            total += inrement
            if progress is not None:
                progress.update(task, advance=inrement, progress=f" {total:4d}")
        # [N, D]
        allFeatures = torch.cat(allFeatures)
        allIdx = torch.cat(allIdx)
        if progress is not None:
            progress.remove_task(task)
        return allIdx, torch.from_numpy(self.database.search(allFeatures.numpy(), numReturns))
