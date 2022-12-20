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

from modfire.dataset import DatasetRegistry, DataPipeRegistry
from modfire.model import ModelRegistry
from modfire.criterion import CriterionRegistry
from modfire.train.hooks import HookRegistry
import modfire.train.schedulers

SchdrRegistry = modfire.train.schedulers.SchdrRegistry
ValueRegistry = modfire.train.values.ValueRegistry

__all__ = [
    "ValueRegistry",
    "SchdrRegistry",
    "ModelRegistry",
    "CriterionRegistry",
    "OptimRegistry",
    "HookRegistry",
    "DatasetRegistry",
    "DataPipeRegistry"
]

class OptimRegistry(Registry[Callable[..., Optimizer]]):
    pass


OptimRegistry.register("Adam")(Adam)
OptimRegistry.register("SGD")(SGD)
if FusedLAMB is not None:
    OptimRegistry.register("Lamb")(FusedLAMB)
