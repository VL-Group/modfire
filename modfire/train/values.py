from typing import Callable

from vlutils.base import Registry

from modfire.utils import ValueBase


class ValueRegistry(Registry[Callable[..., ValueBase]]):
    pass


@ValueRegistry.register
class Constant(ValueBase):
    pass

@ValueRegistry.register
class Exponetial(ValueBase):
    def __init__(self, initValue: float, gamma:float):
        super().__init__(initValue)
        self._gamma = gamma

    def calc(self):
        return self._initValue * (self._gamma ** self._epoch)
