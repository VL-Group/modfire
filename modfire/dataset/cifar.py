import abc
from typing import Iterator, Any, Union, Tuple

from PIL import Image
import torch
import numpy as np
from torch.utils.data import IterDataPipe, default_collate
from torch.utils.data.datapipes.iter import IterableWrapper
from torchvision.datasets import CIFAR10 as _c10, CIFAR100 as _c100

from .dataset import Database, Dataset, TrainSet, QuerySet
from .utils import TrainTransform, EvalTransform


def trainTransform(inputs):
    return TrainTransform(Image.fromarray(inputs[0])), inputs[1]

def evalTransform(inputs):
    return inputs[0], EvalTransform(Image.fromarray(inputs[1]))

class CIFAR(Dataset, abc.ABC):
    def __init__(self, root: str, batchSize: int):
        super().__init__()
        allImages, allTargets = self.getAlldata(root, True)
        allLabels = torch.unique(allTargets)
        allTrains, allQueries, allDatabase = list(), list(), list()
        allTrainLabels, allQueryLabels, allDatabaseLabels = list(), list(), list()
        for label in allLabels:
            images = allImages[label == allTargets]
            targets = allTargets[label == allTargets]
            # 1000, 100, 5900 or 100, 10, 590
            trainSize, querySize = len(images) // 6, len(images) // 60
            train, query, database = images[:trainSize], images[trainSize:(trainSize + querySize)], images[(trainSize + querySize):]
            trainLabel, queryLabel, databaseLabel = targets[:trainSize], targets[trainSize:(trainSize + querySize)], targets[(trainSize + querySize):]
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
        self.batchSize = batchSize
        # self.trainTransform = trainTransform
        # self.evalTransform = evalTransform
        # self.targetTransform = targetTransform

    @staticmethod
    @abc.abstractmethod
    def getAlldata(root, shuffle: bool = True):
        raise NotImplementedError
        # train, test = _c10(root=root, train=True), _c10(root=root, train=False)
        # allImages = np.concatenate((train.data, test.data))
        # allTargets = np.concatenate((np.array(train.targets), np.array(test.targets)))
        # return allImages, allTargets

    @property
    def TrainSet(self) -> TrainSet:
        # class _dataPipe(IterDataPipe):
        #     trains = self.allTrains.numpy()
        #     labels = self.allTrainLabels
        #     # transform = self.trainTransform
        #     # target_transform = self.targetTransform
        #     def __iter__(self):
        #         for img, target in zip(self.trains, self.labels):
        #             # doing this so that it is consistent with all other datasets
        #             # to return a PIL Image
        #             img = Image.fromarray(img)

        #             # if self.transform is not None:
        #             #     img = self.transform(img)

        #             # if self.target_transform is not None:
        #             #     target = self.target_transform(target)
        #             yield img, target
        class _trainSet(TrainSet):
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
                return IterableWrapper(zip(self._trains, self._labels)).shuffle().sharding_filter().map(trainTransform).batch(self._batchSize).collate()
        return _trainSet()

    @property
    def QuerySet(self) -> QuerySet:
        # class _dataPipe(IterDataPipe):
        #     queries = self.allQueries.numpy()
        #     # transform = self.evalTransform
        #     def __iter__(self) -> Iterator[Union[int, torch.Tensor]]:
        #         for i, img in enumerate(self.queries):
        #             # doing this so that it is consistent with all other datasets
        #             # to return a PIL Image
        #             img = Image.fromarray(img)

        #             # if self.transform is not None:
        #             #     img = self.transform(img)
        #             yield i, img

        class _querySet(QuerySet):
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
                # return self._pipe.sharding_filter().map(lambda idxImg: (idxImg[0], EvalTransform(idxImg[1]))).batch(self._batchSize)
                return IterableWrapper(zip(range(len(self._queries)), self._queries)).sharding_filter().map(evalTransform).batch(self._batchSize).collate()
            def info(self) -> torch.Tensor:
                return self._allQueryLabels
        return _querySet()

    @property
    def Database(self) -> Database:
        # class _dataPipe(IterDataPipe):
        #     database = self.allDatabase.numpy()
        #     # transform = self.evalTransform
        #     def __iter__(self) -> Iterator[Union[int, torch.Tensor]]:
        #         for i, img in enumerate(self.database):
        #             # doing this so that it is consistent with all other datasets
        #             # to return a PIL Image
        #             img = Image.fromarray(img)

        #             # if self.transform is not None:
        #             #     img = self.transform(img)
        #             yield i, img

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
                # return self._pipe.sharding_filter().map(lambda idxImg: (idxImg[0], EvalTransform(idxImg[1]))).batch(self._batchSize)
                return IterableWrapper(zip(range(len(self._database)), self._database)).sharding_filter().map(evalTransform).batch(self._batchSize).collate()
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
                # [Nq, k]
                databaseLabels = self._baseLabels[rankList]
                #           [Nq, 1]
                matching = queryInfo[:, None] == databaseLabels
                # [Nq, Nk] -> [Nq]
                numAllTrues = (queryInfo[:, None] == self._baseLabels).sum(-1)
                return matching, numAllTrues
        return _database()


class CIFAR10(CIFAR):
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
        return allImages, allTargets


class CIFAR100(CIFAR):
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
        return allImages, allTargets
