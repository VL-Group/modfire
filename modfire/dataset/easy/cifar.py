import abc
from typing import Any, Tuple, List

from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper
from torchvision.datasets import CIFAR10 as _c10, CIFAR100 as _c100
from vlutils.saver import StrPath

from modfire.dataset import Database, Dataset, TrainSplit, QuerySplit
from modfire.dataset.utils import defaultTrainingDataPipe, defaultEvalDataPipe



class CIFAR(Dataset, abc.ABC):
    def __init__(self, root: StrPath, mode: str, batchSize: int):
        super().__init__(root, mode, batchSize)
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

    @staticmethod
    @abc.abstractmethod
    def getAlldata(root, shuffle: bool = True):
        raise NotImplementedError

    @property
    def TrainSplit(self) -> TrainSplit:
        _ = super().TrainSplit
        class _trainSet(TrainSplit):
            _len = len(self.allTrains)
            _batchSize = self.batchSize
            _trains = self.allTrains.numpy()
            _labels = self.allTrainLabels
            def __len__(self):
                return self._len
            @property
            def BatchSize(self) -> int:
                return self._batchSize
            @property
            def DataPipe(self) -> IterDataPipe:
                return defaultTrainingDataPipe(IterableWrapper(self._trains).map(Image.fromarray).zip(IterableWrapper(self._labels)), self._batchSize)
        return _trainSet()

    @property
    def QuerySplit(self) -> QuerySplit:
        _ = super().QuerySplit
        class _querySet(QuerySplit):
            # _pipe = _dataPipe()
            _allQueryLabels = self.allQueryLabels
            _len = len(self.allQueries)
            _batchSize = self.batchSize
            _queries = self.allQueries.numpy()

            def __len__(self):
                return self._len
            @property
            def BatchSize(self) -> int:
                return self._batchSize
            @property
            def DataPipe(self) -> IterDataPipe:
                return defaultEvalDataPipe(IterableWrapper(self._queries).enumerate(), self._batchSize)
            def info(self, indices: torch.Tensor) -> torch.Tensor:
                # [Nq, nClass]
                return self._allQueryLabels[indices]
        return _querySet()

    @property
    def Database(self) -> Database:
        _ = super().Database
        class _database(Database):
            # _pipe = _dataPipe()
            _baseLabels = self.allDatabaseLabels
            _len = len(self.allDatabase)
            _batchSize = self.batchSize
            _database = self.allDatabase.numpy()
            def __len__(self):
                return self._len
            @property
            def BatchSize(self) -> int:
                return self._batchSize
            @property
            def DataPipe(self):
                return defaultEvalDataPipe(IterableWrapper(self._database).enumerate(), self._batchSize)
            def judge(self, queryInfo: Any, rankList: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                """Return true positives based on query info.

                Args:
                    queryInfo (torch.Tensor): Numeric label of [Nq] tensor. Labels follow CIFAR-10 / CIFAR-100.
                    rankList (torch.Tensor): [Nq, numReturns] indices, each row represents a rank list of top K from database. Indices are obtained by DatPipe.

                Returns:
                    torch.Tensor: [Nq, numReturns] true positives.
                    torch.Tensor: [Nq], Number of all trues w.r.t. query.
                """
                # NOTE: Here, queryInfo is label of queries.
                # [Nq, k, nClass]
                databaseLabels = self._baseLabels[rankList]
                #                            [Nq, 1, nClass]
                matching = torch.logical_and(queryInfo[:, None], databaseLabels).sum(-1) > 0
                # [Nq, Nb, nclass] -> [Nq]
                numAllTrues = (torch.logical_and(queryInfo[:, None], self._baseLabels).sum(-1) > 0).sum(-1)
                return matching, numAllTrues
        return _database()


class CIFAR10(CIFAR):
    def check(self) -> bool:
        try:
            _c10(root=self.root, train=True)
        except RuntimeError:
            return False
        return True

    def prepare(self) -> bool:
        try:
            _c10(root=self.root, download=True)
        except RuntimeError:
            return False
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
        return allImages, F.one_hot(allTargets, num_classes=10) > 0

    @property
    def Semantics(self) -> List[str]:
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class CIFAR100(CIFAR):
    def check(self) -> bool:
        try:
            _c100(root=self.root, train=True)
        except RuntimeError:
            return False
        return True

    def prepare(self) -> bool:
        try:
            _c100(root=self.root, download=True)
        except RuntimeError:
            return False
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
        return allImages, F.one_hot(allTargets, num_classes=100) > 0

    @property
    def Semantics(self) -> List[str]:
        return ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"]
