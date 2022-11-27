import abc
from typing import Iterator, Any, Union, Tuple

from PIL import Image
import torch
import numpy as np
from torch.utils.data import MapDataPipe, IterDataPipe
from torchvision.datasets import CIFAR10 as _c10, CIFAR100 as _c100

from .dataset import Database, Dataset, TrainSet, QuerySet

class CIFAR(Dataset, abc.ABC):
    def __init__(self, root, trainTransform, evalTransform, targetTransform):
        super().__init__()
        allImages, allTargets = self.getAlldata(root, True)
        allLabels = torch.unique(allTargets)
        allTrains, allQueries, allDatabase = list(), list(), list()
        allTrainLabels, allQueryLabels, allDatabaseLabels = list(), list(), list()
        for label in allLabels:
            images = allImages[label == allTargets]
            targets = allTargets[label == allTargets]
            # 1000, 100, 5900 or 100, 10, 590
            trainSize, querySize = images // 6, images // 60
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
        self.trainTransform = trainTransform
        self.evalTransform = evalTransform
        self.targetTransform = targetTransform

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
        class _dataPipe(MapDataPipe):
            trains = self.allTrains
            labels = self.allTrainLabels
            transform = self.trainTransform
            target_transform = self.targetTransform
            def __getitem__(self, index):
                img, target = self.data[index], self.targets[index]

                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                img = Image.fromarray(img)

                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)
                return img, target

            def __len__(self):
                return len(self.trains)
        class _trainSet(TrainSet):
            _pipe = _dataPipe()
            def DataPipe(self) -> MapDataPipe:
                return self._pipe
        return _trainSet()

    @property
    def QuerySet(self) -> QuerySet:
        class _dataPipe(IterDataPipe):
            queries = self.allQueries
            transform = self.evalTransform
            def __iter__(self) -> Iterator[Union[int, torch.Tensor]]:
                for i, img in enumerate(self.queries):
                    # doing this so that it is consistent with all other datasets
                    # to return a PIL Image
                    img = Image.fromarray(img)

                    if self.transform is not None:
                        img = self.transform(img)
                    yield i, img

            def __len__(self):
                return len(self.database)

        class _querySet(QuerySet):
            _pipe = _dataPipe()
            _allQueryLabels = self.allQueryLabels
            def DataPipe(self) -> IterDataPipe:
                return self._pipe
            def info(self) -> torch.Tensor:
                return self._allQueryLabels
        return _querySet()

    def _baseSplit(self) -> IterDataPipe:
        class _dataPipe(IterDataPipe):
            database = self.allDatabase
            transform = self.evalTransform
            def __iter__(self) -> Iterator[Union[int, torch.Tensor]]:
                for i, img in enumerate(self.database):
                    # doing this so that it is consistent with all other datasets
                    # to return a PIL Image
                    img = Image.fromarray(img)

                    if self.transform is not None:
                        img = self.transform(img)
                    yield i, img

            def __len__(self):
                return len(self.database)

        return _dataPipe()

    @property
    def Database(self) -> Database:
        class _database(Database):
            _dataPipe = self._baseSplit()
            _baseLabels = self.allDatabaseLabels
            @property
            def DataPipe(self):
                return self._dataPipe
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
    @abc.abstractmethod
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
    @abc.abstractmethod
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
