

if len(allConcepts) != 81:
    raise ValueError("NUS-WIDE concept list corrupted.")

from typing import Any, Tuple

from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper
from torchvision.datasets import CIFAR10 as _c10, CIFAR100 as _c100
from vlutils.saver import StrPath

from modfire.dataset import Database, Dataset, TrainSet, QuerySet
from modfire.dataset.utils import TrainTransform, EvalTransform


_FILE_PATH = ""
_GLOBAL_HASH = "71bf207c"
_ALL_CONCEPTS = ["airport", "animal", "beach", "bear", "birds", "boats", "book", "bridge", "buildings", "cars", "castle", "cat", "cityscape", "clouds", "computer", "coral", "cow", "dancing", "dog", "earthquake", "elk", "fire", "fish", "flags", "flowers", "food", "fox", "frost", "garden", "glacier", "grass", "harbor", "horses", "house", "lake", "leaf", "map", "military", "moon", "mountain", "nighttime", "ocean", "person", "plane", "plants", "police", "protest", "railroad", "rainbow", "reflection", "road", "rocks", "running", "sand", "sign", "sky", "snow", "soccer", "sports", "statue", "street", "sun", "sunset", "surf", "swimmers", "tattoo", "temple", "tiger", "tower", "town", "toy", "train", "tree", "valley", "vehicle", "water", "waterfall", "wedding", "whales", "window", "zebra"]


class NUS_WIDE(Dataset):
    def __init__(self, root: StrPath):
        super().__init__(root)

    def check(self) -> bool:
        return False

    def prepare(self, root: StrPath):
        return super().prepare(root)
