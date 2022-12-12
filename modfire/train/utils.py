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



def setWeightDecay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    # remove duplicates
    cleanedParams = {}
    paramIds = set()
    for key in params:
        cleanedParams[key] = list()
        for x in params[key]:
            if id(x) not in paramIds:
                cleanedParams[key].append(x)
                paramIds.add(id(x))

    param_groups = []
    for key in cleanedParams:
        if len(cleanedParams[key]) > 0:
            param_groups.append({"params": cleanedParams[key], "weight_decay": params_weight_decay[key]})
    return param_groups
