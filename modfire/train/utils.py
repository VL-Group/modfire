from typing import Union, Optional, Any, List, Tuple

import torch
from torch import nn
from vlutils.saver import Saver, DummySaver, StrPath
from rich import filesize



def getSaver(saveDir: StrPath, saveName: StrPath = "saved.ckpt", loggerName: str = "root", loggingLevel: Union[str, int] = "INFO", config: Any = None, autoManage: bool = True, maxItems: int = 25, reserve: bool = False, dumpFile: Optional[str] = None, activateTensorboard: bool = True, disable: bool = False):
    if disable:
        return DummySaver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)
    else:
        return Saver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)

getSaver.__doc__ = Saver.__doc__


def formatStep(step):
    unit, suffix = filesize.pick_unit_and_suffix(step, ["", "k", "M"], 1000)
    if unit < 10:
        return f"{(step // unit):5d}"
    else:
        truncated = step / unit
        if truncated < 10:
            return f"{truncated:4.6f}"[:4] + suffix
        elif truncated < 100:
            return f"{truncated:4.6f}"[:4] + suffix
        else:
            return f"{truncated:11.6f}"[:4] + suffix


class PrettyStep:
    def __init__(self):
        self._lastFormatted = -1
        self._prettyStep = "......"

    def __call__(self, step) -> str:
        if step == self._lastFormatted:
            return self._prettyStep
        else:
            self._prettyStep = formatStep(step)
            self._lastFormatted = step
            return self._prettyStep

class EMATracker(nn.Module):
    def __init__(self, size: Union[torch.Size, List[int], Tuple[int, ...]], momentum: float = 0.9):
        super().__init__()
        self._shadow: torch.Tensor
        self._decay = 1 - momentum
        self.register_buffer("_shadow", torch.empty(size) * torch.nan)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if torch.all(torch.isnan(self._shadow)):
            self._shadow.copy_(x)
            return self._shadow
        self._shadow -= self._decay * (self._shadow - x)
        return self._shadow
