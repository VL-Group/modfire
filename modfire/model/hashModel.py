from abc import ABC, abstractmethod
import numbers
from typing import Union

from torch import nn, Tensor
from torchvision.models import get_model, get_model_weights
from vlutils.base import Registry

from modfire.utils import ValueBase, registry

from .utils import findLastLinear, replaceModule
from .base import BinaryWrapper, ModelRegistry


_PRETRAINED_MODEL_CLASSES = 1000


class Backbone(nn.Module):
    def __init__(self, bits: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self._backbone = get_model(backbone, weights=get_model_weights(backbone).DEFAULT if pretrained else None)
        # modifying backbone
        lastLinears = findLastLinear(self._backbone, _PRETRAINED_MODEL_CLASSES)
        replaceModule(self._backbone, [(name, nn.Linear(linear.in_features, bits)) for name, linear in lastLinears])

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._backbone(x)
        if isinstance(outputs, tuple):
            # google-style networks, ignore aux_logits
            return outputs[0]
        else:
            return outputs


class HashRegistry(Registry):
    pass


class HashLayer(ABC, nn.Module):
    def __init__(self, temperature: Union[dict, numbers.Number] = 1.0):
        super().__init__()
        if isinstance(temperature, numbers.Number):
            temperatureTuner = registry.ValueRegistry.get("Constant")(float(temperature))
        else:
            temperatureTuner = registry.ValueRegistry.get(temperature["key"])(temperature["params"])
        self.temperature = temperatureTuner

    def step(self):
        self.temperature.step()

    @abstractmethod
    def trainableHashFunction(self, h: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, h: Tensor, *args, **kwargs) -> Tensor:
        if self.training:
            return self.trainableHashFunction(h, *args, **kwargs, temperature=self.temperature.Value)
        else:
            return h > 0


@HashRegistry.register
class STEHash(HashLayer):
    def trainableHashFunction(self, h: Tensor, *_, **__) -> Tensor:
        return (h.sign() - h).detach() + h


@HashRegistry.register
class SoftHash(HashLayer):
    def trainableHashFunction(self, h: Tensor, temperature: float = 1.0, *_, **__) -> Tensor:
        return (h / temperature).tanh()


@HashRegistry.register
class LogitHash(HashLayer):
    def trainableHashFunction(self, h: Tensor, temperature: float = 1.0, *_, **__) -> Tensor:
        return h / temperature


@ModelRegistry.register
class HashModel(BinaryWrapper):
    def __init__(self, bits: int, backbone: str, hashMethod: dict, pretrained: bool = True, *_, **__):
        super().__init__(bits)
        self._backboneName = backbone
        self._backbone = Backbone(bits, backbone, pretrained)
        self._hashMethod = HashRegistry.get(hashMethod["key"])(**hashMethod.get("params", {}))

    def step(self):
        self._hashMethod.step()

    @property
    def Temperature(self) -> float:
        return self._hashMethod.temperature.Value

    def forward(self, x, *args, **kwArgs):
        z = self._backbone(x)
        return { "z": z, "b": self._hashMethod(z, *args, **kwArgs) }

    def encode(self, image: Tensor):
        z = self._backbone(image)
        return self.boolToByte(z > 0)

    def summary(self) -> str:
        return "_".join(map(str, [self.Type, self._backboneName, self.bits]))
