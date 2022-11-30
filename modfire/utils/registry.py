from warnings import warn
from typing import Callable,Any

from torch.optim import Adam, SGD, Optimizer
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
import modfire.train.schedulers

SchdrRegistry = modfire.train.schedulers.SchdrRegistry

__all__ = [
    "SchdrRegistry",
    "ModelRegistry",
    "CriterionRegistry",
    "OptimRegistry",
    "HookRegistry",
    "FunctionRegistry",
    "DatasetRegistry"
]

class OptimRegistry(Registry[Callable[..., Optimizer]]):
    pass


class HookRegistry(Registry[Any]):
    pass

class FunctionRegistry(Registry[Callable]):
    pass


OptimRegistry.register("Adam")(Adam)
OptimRegistry.register("SGD")(SGD)
if FusedLAMB is not None:
    OptimRegistry.register("Lamb")(FusedLAMB)
