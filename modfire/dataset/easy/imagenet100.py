
from typing import List
import os
import logging
import tarfile
import glob
import shutil

import torch
import numpy as np
from vlutils.saver import StrPath

from ..dataset import Database, Dataset, TrainSplit, QuerySplit, DatasetRegistry
from modfire import Consts
from modfire.utils import concatOfFiles, hashOfFile, hashOfStream, getRichProgress


_ASSETS_PATH = Consts.AssetsPath.joinpath("imagenet")
_FILE_URL = [
    "https://github.com/VL-Group/modfire/releases/download/ImageNet/ImageNet_29ab4c9d.tar.gz.0",
    "https://github.com/VL-Group/modfire/releases/download/ImageNet/ImageNet_4daa42d2.tar.gz.1",
    "https://github.com/VL-Group/modfire/releases/download/ImageNet/ImageNet_b27acacf.tar.gz.2",
    "https://github.com/VL-Group/modfire/releases/download/ImageNet/ImageNet_bcd5f0b0.tar.gz.3",
    "https://github.com/VL-Group/modfire/releases/download/ImageNet/ImageNet_97feda8e.tar.gz.4",
    "https://github.com/VL-Group/modfire/releases/download/ImageNet/ImageNet_f617cd93.tar.gz.6",
    "https://github.com/VL-Group/modfire/releases/download/ImageNet/ImageNet_e96bd3ee.tar.gz.7",
    "https://github.com/VL-Group/modfire/releases/download/ImageNet/ImageNet_cfbe728c.tar.gz.5",
]
_GLOBAL_HASH = "9514798e"
_FILE_COUNT = 127100 + 5000


@DatasetRegistry.register
class ImageNet100(Dataset):
    def __init__(self, root: StrPath, mode: str, batchSize: int, pipeline):
        super().__init__(root, mode, batchSize, pipeline)
        self._allImages, self._allLabels = self.readImageList(os.path.join(_ASSETS_PATH, self.mode.TxtConst))

    def readImageList(self, path: StrPath):
        def _parse(oneLine: str):
            return oneLine.split(maxsplit=1)
        with open(path, "r") as fp:
            allLines = filter(None, (line.strip() for line in fp.readlines()))
        allImages, allLabels = list(), list()
        for line in allLines:
            img, label = _parse(line)
            allImages.append(os.path.join(self.root, img))
            allLabels.append(int(label))
        allLabels = torch.from_numpy(np.array(allLabels))
        if len(allLabels.shape) != 1 or len(torch.unique(allLabels)) != 100:
            raise ValueError(f"File corrupted. labels have shape: {allLabels.shape} and unique label count: {len(torch.unique(allLabels))} (expect 100).")
        return allImages, torch.nn.functional.one_hot(allLabels, num_classes=100).float()

    def check(self) -> bool:
        return os.path.exists(self.root) and os.path.isdir(self.root) and len(glob.glob(os.path.join(self.root, "**/*.JPEG"), recursive=True)) == _FILE_COUNT

    @staticmethod
    def prepare(root: StrPath, logger = logging) -> bool:
        if os.path.exists(root) and os.path.isdir(root) and len(glob.glob(os.path.join(root, "**/*.JPEG"), recursive=True)) == _FILE_COUNT:
            logger.info("File already prepared, exit.")

            # clean up

            chunkedFiles = glob.glob(os.path.join(root, "*.tar.gz.*")) + glob.glob(os.path.join(root, "tmp*"))

            for f in chunkedFiles:
                os.remove(f)
            return True

        os.makedirs(root, exist_ok=True)

        # clean up

        tmpFiles = glob.glob(os.path.join(root, "tmp*"))
        for f in tmpFiles:
            os.remove(f)


        logger.info("Download files into `%s`.", root)
        logger.warning("To prepare `%s`, you need at least 6 GiB disk space for downloading and extracting.", ImageNet100.__name__)
        for url in _FILE_URL:
            fileName = url.split("/")[-1]
            filePath = os.path.join(root, fileName)
            hashPrefix = url.split("_")[-1].split(".")[0]
            if os.path.exists(filePath):
                with getRichProgress() as p:
                    hashFile = hashOfFile(filePath, p)
                if hashFile.startswith(hashPrefix):
                    logger.info("Skipping `%s` since it is already downloded.", filePath)
                    continue
                else:
                    logger.info("Removing corrupted `%s`.", filePath)
                    os.remove(filePath)
            torch.hub.download_url_to_file(url, root, hashPrefix)


        logger.info("Verifying...")

        chunkedFiles = glob.glob(os.path.join(root, "*.tar.gz.*"))
        if len(chunkedFiles) != len(_FILE_URL):
            raise ValueError(f"Find incorrect downloaded files. File list is {chunkedFiles}.")
        with concatOfFiles(sorted(chunkedFiles, key=lambda x: int(x.split(".")[-1]))) as stream:
            hashValue = hashOfStream(stream)
        if not hashValue.startswith(_GLOBAL_HASH):
            raise BufferError("Merged file corrupted, please try again.")

        logger.info("Verifyied.")

        logger.info("Extracting...")
        extratedPath = os.path.join(root, "temp")
        with concatOfFiles(sorted(chunkedFiles, key=lambda x: int(x.split(".")[-1]))) as stream:
            tar = tarfile.open(mode="r:gz", fileobj=stream)
            tar.extractall(extratedPath)
        logger.info("Extracted.")


        # clean up

        # chunkedFiles = glob.glob(os.path.join(root, "*.tar.gz.*")) + glob.glob(os.path.join(root, "tmp*"))

        # for f in chunkedFiles:
        #     os.remove(f)

        with getRichProgress() as p:
            src = glob.glob(os.path.join(extratedPath, "*"))
            task = p.add_task("[ Clean up ]", total=len(src), progress="0.00%", suffix="")
            for i, img in enumerate(src):
                shutil.move(img, root)
                p.update(task, advance=1, progress=f"{i / len(src) * 100 :.2f}%")
            p.remove_task(task)
        shutil.rmtree(extratedPath)

        allImagesCount = len(glob.glob(os.path.join(root, "**/*.JPEG"), recursive=True))

        if allImagesCount != _FILE_COUNT:
            raise ValueError(f"The total count of extracted images is incorrect. Expected: {_FILE_COUNT}. Got: {allImagesCount}.")
        return True

    @property
    def TrainSplit(self) -> TrainSplit:
        class _dataset(TrainSplit):
            @property
            def NumClass(self) -> int:
                return 100

        return _dataset(self._allImages, self._allLabels, self.batchSize, self._loadImg, self.pipeline)

    @property
    def QuerySplit(self) -> QuerySplit:
        return QuerySplit(self._allImages, self._allLabels, self.batchSize, self._loadImg, self.pipeline)

    @property
    def Database(self) -> Database:
        return Database(self._allImages, self._allLabels, self.batchSize, self._loadImg, self.pipeline)

    @property
    def Semantics(self) -> List[str]:
        import json
        import warnings
        warnings.warn("We use the simplified labels from `https://github.com/anishathalye/imagenet-simple-labels`. Labels are different from the original one.")
        with open(os.path.join(_ASSETS_PATH, "labels.json"), "r") as fp:
            # a 1000 length list
            allLabels = json.load(fp)
        with open(os.path.join(_ASSETS_PATH, "imagenet100_class_map.txt"), "r") as fp:
            mapper = map(int, filter(None, (line.strip() for line in fp.readlines())))
        return [allLabels[i] for i in mapper]
