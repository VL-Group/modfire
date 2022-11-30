from typing import Callable

from vlutils.base import Registry
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR


class SchdrRegistry(Registry[Callable[..., _LRScheduler]]):
    pass

@SchdrRegistry.register("CosineAnnealingLRWarmUp")
def cosineAnnealingLRWarmUp(optimizer, epochs, warmupEpochs, warmupDecay):
    warmupScheduler = LinearLR(optimizer, start_factor=warmupDecay, total_iters=warmupEpochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmupEpochs)
    return SequentialLR(optimizer, [warmupScheduler, scheduler], milestones=[warmupEpochs])

SchdrRegistry.register("ExponentialLR")(ExponentialLR)
SchdrRegistry.register("CosineAnnealingWarmRestarts")(CosineAnnealingWarmRestarts)
SchdrRegistry.register("CosineAnnealingLR")(CosineAnnealingLR)
SchdrRegistry.register("ReduceLROnPlateau")(ReduceLROnPlateau)
