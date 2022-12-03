
from typing import Any, Tuple, List
import os
import logging
import tarfile
import glob

import torch
import numpy as np
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper
from torchvision.datasets import folder
from vlutils.saver import StrPath

from modfire.dataset import Database, Dataset, TrainSplit, QuerySplit
from modfire.dataset.utils import defaultEvalDataPipe, defaultTrainingDataPipe
from modfire import Consts
from modfire.utils import concatOfFiles


_ASSETS_PATH = Consts.AssetsPath.joinpath("nuswide")
_FILE_URL = [
    "https://github.com/VL-Group/modfire/releases/download/nus-wide/NUS-WIDE_cd909bce.tar.gz.0",
    "https://github.com/VL-Group/modfire/releases/download/nus-wide/NUS-WIDE_873c354e.tar.gz.1",
    "https://github.com/VL-Group/modfire/releases/download/nus-wide/NUS-WIDE_529bd7dd.tar.gz.2"
]
_GLOBAL_HASH = "71bf207c"
_FILE_COUNT = 269648
ALL_CONCEPTS = ["airport", "animal", "beach", "bear", "birds", "boats", "book", "bridge", "buildings", "cars", "castle", "cat", "cityscape", "clouds", "computer", "coral", "cow", "dancing", "dog", "earthquake", "elk", "fire", "fish", "flags", "flowers", "food", "fox", "frost", "garden", "glacier", "grass", "harbor", "horses", "house", "lake", "leaf", "map", "military", "moon", "mountain", "nighttime", "ocean", "person", "plane", "plants", "police", "protest", "railroad", "rainbow", "reflection", "road", "rocks", "running", "sand", "sign", "sky", "snow", "soccer", "sports", "statue", "street", "sun", "sunset", "surf", "swimmers", "tattoo", "temple", "tiger", "tower", "town", "toy", "train", "tree", "valley", "vehicle", "water", "waterfall", "wedding", "whales", "window", "zebra"]


if len(ALL_CONCEPTS) != 81:
    raise ValueError("NUS-WIDE concept list corrupted.")


class NUS_WIDE(Dataset):
    def __init__(self, root: StrPath, mode: str, batchSize: int):
        super().__init__(root, mode, batchSize)
        self._allImages, self._allLabels = self.readImageList(os.path.join(_ASSETS_PATH, self.mode.TxtConst))

    def readImageList(self, path: StrPath):
        def _parse(oneLine: str):
            return oneLine.split(maxsplit=1)
        with open(path, "r") as fp:
            allLines = filter(None, (line.strip() for line in fp.readlines()))
        allImages, allLabels = list()
        for line in allLines:
            img, labels = _parse(line)
            allImages.append(img)
            allLabels.append(map(int, labels.split()))
        allLabels = torch.from_numpy(np.array(allLabels))
        if allLabels.shape != 2 or allLabels.shape[-1] != len(ALL_CONCEPTS):
            raise ValueError(f"File corrupted. labels have shape: {allLabels.shape}, while expected concepts length is {len(ALL_CONCEPTS)}.")
        return allImages, allLabels

    def check(self) -> bool:
        return os.path.exists(self.root) and os.path.isdir(self.root) and len(os.listdir(self.root)) == _FILE_COUNT

    def prepare(self, root: StrPath, logger = logging) -> bool:
        os.makedirs(root, exist_ok=True)

        logger.info("Download files into `%s`.", root)
        logger.warning("To prepare NUS-WIDE, you need at least 10 GiB disk space for downloading and extracting.")
        for url in _FILE_URL:
            torch.hub.download_url_to_file(url, root, url.split("_")[-1].split(".")[0])
        logger.info("Extracting...")
        chunkedFiles = glob.glob(os.path.join(root, "*.tar.gz.*"))
        if len(chunkedFiles) != 3:
            raise ValueError(f"Find incorrect downloaded files. File list is {chunkedFiles}.")
        with concatOfFiles(sorted(chunkedFiles, key=lambda x: int(x.split(".")[-1]))) as stream:
            tar = tarfile.open(mode="r:gz", fileobj=stream)
            tar.extractall()
        logger.info("Extracted.")
        map(os.remove, chunkedFiles)
        if len(os.listdir(root)) != _FILE_COUNT:
            raise ValueError(f"The total count of extracted images is incorrect. Expected: {_FILE_COUNT}. Got: {len(os.listdir(root))}.")
        return True

    @property
    def TrainSplit(self) -> TrainSplit:
        _ = super().TrainSplit
        class _dataset(TrainSplit):
            _images = self._allImages
            _labels = self._allLabels
            _batchSize = self.batchSize
            @property
            def DataPipe(self) -> IterDataPipe:
                imgs = IterableWrapper(self._images)
                labels = IterableWrapper(self._labels)
                return defaultTrainingDataPipe(imgs.map(folder.default_loader).zip(labels), self._batchSize)
            def __len__(self):
                return len(self._images)

        return _dataset()

    @property
    def QuerySplit(self) -> QuerySplit:
        _ = super().QuerySplit
        class _dataset(QuerySplit):
            _images = self._allImages
            _labels = self._allLabels
            _batchSize = self.batchSize
            @property
            def DataPipe(self) -> IterDataPipe:
                imgs = IterableWrapper(self._images)
                return defaultEvalDataPipe(imgs.map(folder.default_loader).enumerate(), self._batchSize)
            def __len__(self):
                return len(self._images)
            def info(self, indices) -> torch.Tensor:
                return self._labels[indices]

        return _dataset()

    @property
    def Database(self) -> Database:
        _ = super().Database
        class _dataset(Database):
            _images = self._allImages
            _labels = self._allLabels
            _batchSize = self.batchSize
            @property
            def DataPipe(self) -> IterDataPipe:
                imgs = IterableWrapper(self._images)
                return defaultEvalDataPipe(imgs.map(folder.default_loader).enumerate(), self._batchSize)
            def __len__(self):
                return len(self._images)
            def judge(self, queryInfo: Any, rankList: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                # NOTE: Here, queryInfo is label of queries.
                # [Nq, k, nClass]
                databaseLabels = self._labels[rankList]
                #                            [Nq, 1, nClass]
                matching = torch.logical_and(queryInfo[:, None], databaseLabels).sum(-1) > 0
                # [Nq, Nb, nclass] -> [Nq]
                numAllTrues = (torch.logical_and(queryInfo[:, None], self._labels).sum(-1) > 0).sum(-1)
                return matching, numAllTrues

        return _dataset()

    @property
    def Semantics(self) -> List[str]:
        return ALL_CONCEPTS
