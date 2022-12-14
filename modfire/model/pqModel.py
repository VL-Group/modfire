from typing import Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models import get_model, get_model_weights
from vlutils.base import Registry

from modfire.utils import ValueBase

from .utils import findLastLinear, replaceModule
from .base import PQWrapper, ModelRegistry


_PRETRAINED_MODEL_CLASSES = 1000


class IntraNormalization(nn.Module):
    def __init__(self, m: int):
        super().__init__()
        self._m = m

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x.reshape(*x.shape[:-1], self._m, -1)).reshape(x.shape)

class Backbone(nn.Module):
    def __init__(self, m: int, d: int, intraNormalization: bool = True, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self._backbone = get_model(backbone, weights=get_model_weights(backbone).DEFAULT if pretrained else None)
        # modifying backbone
        lastLinears = findLastLinear(self._backbone, _PRETRAINED_MODEL_CLASSES)
        replaceModule(self._backbone, [(name, nn.Linear(linear.in_features, d)) for name, linear in lastLinears])
        self._finalLayers = nn.ModuleList(nn.Linear(d, d // m) for _ in range(m))
        self._m = m
        self._intraNormalization = IntraNormalization(m) if intraNormalization else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._backbone(x)
        if isinstance(outputs, tuple):
            # google-style networks, ignore aux_logits
            y = outputs[0]
        else:
            y = outputs
        finalOutputs = list()
        # Multi-head projection
        for i in range(self._m):
            # [N, D] -> [N, D // M]
            finalOutputs.append(self._finalLayers[i](y))
        # [N, D]
        return self._intraNormalization(torch.cat(finalOutputs, -1))


class PQRegistry(Registry):
    pass


class PQLayer(ABC, nn.Module):
    codebook: nn.Parameter
    def __init__(self, codebook: nn.Parameter, temperature: ValueBase):
        super().__init__()
        self.codebook = codebook
        self._m, self._k, self._d_m = codebook.shape
        self.temperature = temperature

    def step(self):
        self.temperature.step()

    # NOTE: ALREADY CHECKED CONSISTENCY WITH NAIVE IMPL.
    def _distance(self, x: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        # [n, m, d_m]
        x = x.reshape(n, self._m, self._d_m)

        # [n, m, 1]
        x2 = (x ** 2).sum(-1, keepdim=True)

        # [m, k]
        c2 = (self.codebook ** 2).sum(-1)
        # [n, m, d] * [m, k, d] -sum-> [n, m, k]
        inter = torch.einsum("nmd,mkd->nmk", x, self.codebook)
        # [n, m, k]
        distance = x2 + c2 - 2 * inter
        return distance

    @abstractmethod
    def trainablePQFunction(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        if self.training:
            # NOTE: Due to the asymmertric distance computation, we need to return x and q simultaneously for training.
            return self.trainablePQFunction(x, *args, **kwargs, temperature=self.temperature.Value)
        else:
            # ** IMPORTANT **: The real quantization happens on the inner PQDatabase in PQModel during indexing.
            raise NotImplementedError


@PQRegistry.register
class SoftPQ(PQLayer):
    def trainablePQFunction(self, x: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
        # [n, m, k]
        distance = self._distance(x)
        # [n, m, k]
        logit = (-distance / temperature).softmax(-1)
        return x, torch.einsum("nmk,mkd->nmd", logit, self.codebook).reshape(x.shape), logit

@PQRegistry.register
class SoftSTEPQ(PQLayer):
    def trainablePQFunction(self, x: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
        # [n, m, k]
        distance = self._distance(x)
        # [n, m, k]
        logit = (-distance / temperature).softmax(-1)
        soft = torch.einsum("nmk,mkd->nmd", logit, self.codebook).reshape(x.shape)
        # [n, m]
        hard = distance.argmin(-1)
        hard = F.one_hot(hard, num_classes=self._k)
        hard = torch.einsum("nmk,mkd->nmd", hard, self.codebook).reshape(x.shape)
        return x, (hard - soft).detach() + soft, logit

@PQRegistry.register
class HardPQ(PQLayer):
    def trainablePQFunction(self, x: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
        # [n, m, k]
        distance = self._distance(x)
        # [n, m]
        hard = distance.argmin(-1)
        hard = F.one_hot(hard, num_classes=self._k)
        hard = torch.einsum("nmk,mkd->nmd", hard, self.codebook).reshape(x.shape)
        return x, (hard - x).detach() + x, (-distance / temperature).softmax(-1)

@PQRegistry.register
class GumbelPQ(PQLayer):
    def __init__(self, codebook: nn.Parameter, temperature: ValueBase):
        super().__init__(codebook, temperature)
        self._temperature = nn.Parameter(torch.ones((self._m, 1)))

    def trainablePQFunction(self, x: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
        # [n, m, k]
        distance = self._distance(x)
        # [n, m, k]
        hard = F.gumbel_softmax(-distance * self._temperature, temperature, hard=True)
        hard = torch.einsum("nmk,mkd->nmd", hard, self.codebook).reshape(x.shape)
        return x, hard, (-distance / temperature).softmax(-1)


@ModelRegistry.register
class PQModel(PQWrapper):
    def __init__(self, m: int, k: int, d: int, intraNormalization: bool, backbone: str, pqMethod: str, *args, **kwArgs):
        super().__init__(m, k, d)
        self._backbone = Backbone(m, d, intraNormalization, backbone)
        self._pqMethod = PQRegistry.get(pqMethod)(self.codebook, *args, **kwArgs)

    def step(self):
        self._pqMethod.step()

    @property
    def Temperature(self) -> float:
        return self._pqMethod.temperature.Value

    def forward(self, x, *args, **kwArgs):
        x = self._backbone(x)
        x, q, l = self._pqMethod(x, *args, **kwArgs)
        return { "x": x, "q": q, "logits": l, "codebook": self.codebook }

    def encode(self, image: Tensor):
        # ** IMPORTANT **: Use database to encode, not here
        return self._backbone(image)
