import abc
from typing import List
import logging

from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import CIFAR10 as _c10, CIFAR100 as _c100
from vlutils.saver import StrPath

from ..dataset import Database, Dataset, TrainSplit, QuerySplit, DatasetRegistry


class CIFAR(Dataset, abc.ABC):
    def __init__(self, root: StrPath, mode: str, batchSize: int, pipeline):
        super().__init__(root, mode, batchSize, pipeline)
        allImages, allTargets = self.getAlldata(root, True)
        allTrains, allQueries, allDatabase = list(), list(), list()
        allTrainLabels, allQueryLabels, allDatabaseLabels = list(), list(), list()
        # 0 ~ nClass
        for i in range(allTargets.shape[-1]):
            images = allImages[i == allTargets.float().argmax(-1)]
            targets = allTargets[i == allTargets.float().argmax(-1)]
            # 1000, 100, 5900 or 100, 10, 590
            trainSize, querySize = len(images) // 6, len(images) // 60
            train, query, database = images[querySize:(querySize + trainSize)], images[:querySize], images[querySize:]
            trainLabel, queryLabel, databaseLabel = targets[querySize:(querySize + trainSize)], targets[:querySize], targets[querySize:]
            allTrains.append(train)
            allQueries.append(query)
            allDatabase.append(database)
            allTrainLabels.append(trainLabel)
            allQueryLabels.append(queryLabel)
            allDatabaseLabels.append(databaseLabel)
        self.allTrains = torch.cat(allTrains)
        self.allQueries = torch.cat(allQueries)
        self.allDatabase = torch.cat(allDatabase)
        self.allTrainLabels = torch.cat(allTrainLabels)
        self.allQueryLabels = torch.cat(allQueryLabels)
        self.allDatabaseLabels = torch.cat(allDatabaseLabels)
        # self.trainTransform = trainTransform
        # self.evalTransform = evalTransform
        # self.targetTransform = targetTransform


    def _loadImg(self, inputs):
        i, img = inputs
        return i, Image.fromarray(img.numpy())

    @staticmethod
    @abc.abstractmethod
    def getAlldata(root, shuffle: bool = True):
        raise NotImplementedError

    @property
    def TrainSplit(self) -> TrainSplit:
        class _trainSet(TrainSplit):
            _numClass = len(self.Semantics)
            @property
            def NumClass(self) -> int:
                return self._numClass
        return _trainSet(self.allTrains, self.allTrainLabels, self.batchSize, self._loadImg, self.pipeline)

    @property
    def QuerySplit(self) -> QuerySplit:
        return QuerySplit(self.allQueries, self.allQueryLabels, self.batchSize, self._loadImg, self.pipeline)

    @property
    def Database(self) -> Database:
        return Database(self.allDatabase, self.allDatabaseLabels, self.batchSize, self._loadImg, self.pipeline)


@DatasetRegistry.register
class CIFAR10(CIFAR):
    def check(self) -> bool:
        try:
            _c10(root=self.root, train=True)
        except RuntimeError:
            return False
        return True

    @staticmethod
    def prepare(root: StrPath, logger = logging) -> bool:
        logger.info("Start to prepare CIFAR-10...")
        _c10(root=root, download=True)
        return True

    @staticmethod
    def getAlldata(root, shuffle: bool = True):
        train, test = _c10(root=root, train=True), _c10(root=root, train=False)
        # [n, c, h, w]
        allImages = torch.from_numpy(np.concatenate((train.data, test.data)))
        # [n]
        allTargets = torch.from_numpy(np.concatenate((np.array(train.targets), np.array(test.targets))))
        if shuffle:
            randIdx = torch.randperm(len(allImages))
            allImages = allImages[randIdx]
            allTargets = allTargets[randIdx]
        return allImages, F.one_hot(allTargets, num_classes=10).float()

    @property
    def Semantics(self) -> List[str]:
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

@DatasetRegistry.register
class CIFAR100(CIFAR):
    def check(self) -> bool:
        try:
            _c100(root=self.root, train=True)
        except RuntimeError:
            return False
        return True

    @staticmethod
    def prepare(root: StrPath, logger = logging) -> bool:
        logger.info("Start to prepare CIFAR-100...")
        _c100(root=root, download=True)
        return True
    @staticmethod
    def getAlldata(root, shuffle: bool = True):
        train, test = _c100(root=root, train=True), _c100(root=root, train=False)
        # [n, c, h, w]
        allImages = torch.from_numpy(np.concatenate((train.data, test.data)))
        # [n]
        allTargets = torch.from_numpy(np.concatenate((np.array(train.targets), np.array(test.targets))))
        if shuffle:
            randIdx = torch.randperm(len(allImages))
            allImages = allImages[randIdx]
            allTargets = allTargets[randIdx]
        return allImages, F.one_hot(allTargets, num_classes=100).float()

    @property
    def Semantics(self) -> List[str]:
        return ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"]
