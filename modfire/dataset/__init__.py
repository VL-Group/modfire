from typing import Callable

from vlutils.base import Registry

from .dataset import Database, TrainSplit, QuerySplit, Dataset, Split
from .easy.cifar import CIFAR10, CIFAR100


class DatasetRegistry(Registry[Callable[..., Dataset]]):
    pass


DatasetRegistry.register(CIFAR10)
DatasetRegistry.register(CIFAR100)
