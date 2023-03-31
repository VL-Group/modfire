import abc
from typing import Optional, Tuple, Callable, Iterator
import enum
import math

from rich.progress import Progress
import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from vlutils.base import Registry

from modfire.dataset import Database, QuerySplit
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
    def __init__(self, bits: int):
        super().__init__()
        self.register_buffer("_dummy", torch.empty([1]), persistent=False)
        self.bits = bits

    @property
    def Bits(self) -> int:
        return self.bits

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
    def search(self, queries: QuerySplit, numReturns: int, progress: Optional[Progress] = None) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """_summary_

        Args:
            queries (QuerySplit): _description_
            numReturns (int): _description_
            progress (Optional[Progress], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Yields:
            Iterator[Tuple[torch.Tensor, torch.Tensor]]: a batch of returned results: (indices of queries [Nq], indices of database of nearest neighbor [Nq, numReturn]).
        """
        raise NotImplementedError


class BinaryWrapper(BaseWrapper):
    _byteTemplate: torch.Tensor
    def __init__(self, bits: int):
        super().__init__(bits)
        self.database = BinarySearcher(bits)
        self.register_buffer("_byteTemplate", torch.tensor([int(2 ** x) for x in range(8)]))

    @property
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

    @torch.inference_mode()
    def add(self, database: Database, progress: Optional[Progress] = None):
        if progress is not None:
            task = progress.add_task(f"[ Index ]", total=len(database), progress=f" {0:.1f}k", suffix="")
        with DataLoader2(database.DataPipe, reading_service=MultiProcessingReadingService(num_workers=min(int(math.sqrt(database.BatchSize)), 16))) as dataLoader, autocast():
            total = 0
            for idx, image in dataLoader:
                # [N, bits]
                h = self.encode(image.to(self._dummy.device, non_blocking=True, memory_format=torch.channels_last))
                self.database.add(h.cpu().numpy(), idx.cpu().numpy())
                increment = len(h)
                total += increment
                if progress is not None:
                    progress.update(task, advance=increment, progress=f" {total/1000:.1f}k")
        if progress is not None:
            progress.remove_task(task)

    @torch.inference_mode()
    def remove(self, ids: torch.Tensor):
        return self.database.remove(ids.numpy())

    def reset(self):
        return self.database.reset()

    @torch.inference_mode()
    def search(self, queries: QuerySplit, numReturns: int, progress: Optional[Progress] = None) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if progress is not None:
            task = progress.add_task(f"[ Query ]", total=len(queries), progress=f" {0:.1f}k", suffix="")
        with DataLoader2(queries.DataPipe, reading_service=MultiProcessingReadingService(num_workers=min(int(math.sqrt(queries.BatchSize)), 16))) as dataLoader, autocast():
            total = 0
            for idx, image in dataLoader:
                # [N, bits]
                h = self.encode(image.to(self._dummy.device, non_blocking=True, memory_format=torch.channels_last))
                increment = len(h)
                total += increment
                if progress is not None:
                    progress.update(task, advance=increment, progress=f" {total/1000:.1f}k")

                yield idx.to(self._dummy.device, non_blocking=True), torch.from_numpy(self.database.search(h.cpu().numpy(), numReturns)).to(idx.device, non_blocking=True).to(self._dummy.device, non_blocking=True)
        if progress is not None:
            progress.remove_task(task)


class PQWrapper(BaseWrapper):
    codebook: nn.Parameter
    def __init__(self, m: int, k: int, d: int):
        super().__init__(m * int(math.log2(k)))
        self.codebook = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(m, k, d // m)))
        self.database = PQSearcher(self.codebook.cpu().detach().numpy())

    @property
    def Type(self) -> ModelType:
        return ModelType.ProductQuantization

    def updateCodebook(self):
        self.database.assignCodebook(self.codebook.cpu().detach().numpy())

    @torch.inference_mode()
    def add(self, database: Database, progress: Optional[Progress] = None):
        if progress is not None:
            task = progress.add_task(f"[ Index ]", total=len(database), progress=f" {0:.1f}k", suffix="")
        with DataLoader2(database.DataPipe, reading_service=MultiProcessingReadingService(num_workers=min(int(math.sqrt(database.BatchSize)), 16))) as dataLoader, autocast():
            total = 0
            for idx, image in dataLoader:
                # [N, D]
                x = self.encode(image.to(self._dummy.device, non_blocking=True, memory_format=torch.channels_last))
                self.database.add(x.cpu().numpy(), idx.cpu().numpy())
                increment = len(x)
                total += increment
                if progress is not None:
                    progress.update(task, advance=increment, progress=f" {total/1000:.1f}k")
        if progress is not None:
            progress.remove_task(task)

    @torch.inference_mode()
    def remove(self, ids: torch.Tensor):
        return self.database.remove(ids.numpy())

    def reset(self):
        self.updateCodebook()
        return self.database.reset()

    @torch.inference_mode()
    def search(self, queries: QuerySplit, numReturns: int, progress: Optional[Progress] = None) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if progress is not None:
            task = progress.add_task(f"[ Query ]", total=len(queries), progress=f" {0:.1f}k", suffix="")
        with DataLoader2(queries.DataPipe, reading_service=MultiProcessingReadingService(num_workers=min(int(math.sqrt(queries.BatchSize)), 16))) as dataLoader, autocast():
            total = 0
            for idx, image in dataLoader:
                # [D]
                x = self.encode(image.to(self._dummy.device, non_blocking=True, memory_format=torch.channels_last))
                increment = len(x)
                total += increment
                if progress is not None:
                    progress.update(task, advance=increment, progress=f" {total/1000:.1f}k")
                yield idx.to(self._dummy.device, non_blocking=True), torch.from_numpy(self.database.search(x.cpu().numpy(), numReturns)).to(idx.device, non_blocking=True).to(self._dummy.device, non_blocking=True)
        if progress is not None:
            progress.remove_task(task)
