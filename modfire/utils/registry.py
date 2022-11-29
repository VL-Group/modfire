from warnings import warn
from typing import Callable,Any

from torch import nn
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR
try:
    from apex.optimizers import FusedLAMB
    FusedLAMB = FusedLAMB
except ImportError:
    FusedLAMB = None
    warn("Couldn't import apex's `FusedLAMB` optimizer, ignored in `OptimRegistry`.")
from vlutils.base import Registry

from modfire.dataset import DatasetRegistry
from modfire.model import ModelRegistry
from modfire.criterion import CriterionRegistry

__all__ = [
    "ModelRegistry",
    "SchdrRegistry",
    "CriterionRegistry",
    "OptimRegistry",
    "CriterionRegistry",
    "HookRegistry",
    "FunctionRegistry",
    "DatasetRegistry"
]

class OptimRegistry(Registry[Callable[..., Optimizer]]):
    pass

class SchdrRegistry(Registry[Callable[..., _LRScheduler]]):
    pass

class HookRegistry(Registry[Any]):
    pass

class FunctionRegistry(Registry[Callable]):
    pass


OptimRegistry.register("Adam")(Adam)
OptimRegistry.register("SGD")(SGD)
if FusedLAMB is not None:
    OptimRegistry.register("Lamb")(FusedLAMB)

SchdrRegistry.register("ExponentialLR")(ExponentialLR)
SchdrRegistry.register("CosineAnnealingWarmRestarts")(CosineAnnealingWarmRestarts)
SchdrRegistry.register("CosineAnnealingLR")(CosineAnnealingLR)
SchdrRegistry.register("ReduceLROnPlateau")(ReduceLROnPlateau)
