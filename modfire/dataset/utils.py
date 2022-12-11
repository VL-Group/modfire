from typing import Iterator, Callable
import functools

import torch
from torch.utils.data import IterDataPipe
import torch.nn.functional as F
import torchvision.transforms as T
from vlutils.base import Registry


_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]

# https://github.com/pytorch/vision/blob/main/references/classification/presets.py
# https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
TrainTransform = T.Compose([
    T.RandomResizedCrop(176, antialias=True),
    T.RandomHorizontalFlip(),
    T.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR, fill=_IMG_MEAN),
    T.ToTensor(),
    T.Normalize(_IMG_MEAN, _IMG_STD, True),
    T.RandomErasing(p=0.1, value="random", inplace=True),
])

EvalTransform = T.Compose([
    T.Resize(232, antialias=True),
    T.CenterCrop(224),
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


def mappedTrainTransform(inputs):
    return inputs[0], TrainTransform(inputs[1])

def mappedEvalTransform(inputs):
    return inputs[0], EvalTransform(inputs[1])



class DataPipeRegistry(Registry[Callable[..., IterDataPipe]]):
    pass


@DataPipeRegistry.register
def DefaultEvalDataPipe(*_, **__):
    return _defaultEvalDataPipe


@DataPipeRegistry.register
def DefaultTrainingDataPipe(*_, **__):
    return _defaultTrainingDataPipe


@DataPipeRegistry.register
def SelfSupervisedTrainingDataPipe(*_, nViews: int, **__):
    return functools.partial(_selfSupervisedTrainingDataPipe, nViews=nViews)








def _defaultEvalDataPipe(source: IterDataPipe, mapFunctionAfterBaseConversion: Callable, batchSize: int):
    return source.sharding_filter().map(mapFunctionAfterBaseConversion).map(mappedEvalTransform).prefetch(batchSize).batch(batchSize).collate()


def _defaultTrainingDataPipe(source: IterDataPipe, mapFunctionAfterBaseConversion: Callable, batchSize: int):
    return source.cycle().shuffle().sharding_filter().map(mapFunctionAfterBaseConversion).map(mappedTrainTransform).prefetch(batchSize).batch(batchSize).collate()


def _selfSupervisedTrainingDataPipe(source: IterDataPipe, mapFunctionAfterBaseConversion: Callable, batchSize: int, nViews: int):
    return source.cycle().shuffle().sharding_filter().map(mapFunctionAfterBaseConversion).map(mappedTrainTransform).prefetch(batchSize).batch(batchSize).collate()



def toDevice(inputs, device):
    i, img = inputs
    return i.to(device, non_blocking=True), img.to(device, non_blocking=True)
