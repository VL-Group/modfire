from typing import Iterator

import torch
from torch import nn
from torch.utils.data import IterDataPipe, MapDataPipe
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchdata.datapipes.iter import SampleMultiplexer, UnZipper


_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]

TrainTransform = T.Compose([
        T.RandomResizedCrop(256, antialias=True),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR, fill=_IMG_MEAN),
        T.ToTensor(),
        T.Normalize(_IMG_MEAN, _IMG_STD, True),
        T.RandomErasing(value="random", inplace=True),
    ])

EvalTransform = T.Compose([
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(_IMG_MEAN, _IMG_STD, True)
])

class ImageTransform(IterDataPipe):
    def __init__(self, source: IterDataPipe, mode: str):
        super().__init__()
        self._source = source
        self._transform = TrainTransform if mode == "train" else EvalTransform

    def __iter__(self) -> Iterator[torch.Tensor]:
        for img in self._source:
            yield self._transform(img)

def imageTransform(source: IterDataPipe, mode: str):
    transform = TrainTransform if mode == "train" else EvalTransform
    return source.map(lambda x: transform(x))

class LabelToOneHot(IterDataPipe):
    def __init__(self, source: IterDataPipe, numClasses: int):
        super().__init__()
        self._source = source
        self._numClasses = numClasses

    def __iter__(self) -> Iterator[torch.Tensor]:
        for label in self._source:
            yield F.one_hot(label, num_classes=self._numClasses)

def labelToOneHot(source: IterDataPipe, numClasses: int):
    return source.map(lambda x: F.one_hot(x, num_classes=numClasses))
