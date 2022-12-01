from typing import Callable

from torch import nn
from vlutils.base import Registry

class CriterionRegistry(Registry[Callable[..., nn.Module]]):
    pass
