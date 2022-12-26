import logging
import warnings
import os
import signal
from typing import Optional, List
from distutils.version import StrictVersion
import hashlib
from time import sleep
import threading
import abc
import io
from io import IOBase, RawIOBase
import math

from vlutils.logger import LoggerBase
from vlutils.custom import RichProgress
from vlutils.saver import StrPath
from vlutils.base import Restorable
from torch import nn
import torch
from rich import filesize
from rich.progress import Progress
from rich.progress import TimeElapsedColumn, BarColumn, TimeRemainingColumn

from modfire import Consts
import modfire
from modfire.config import Config


def checkConfigSummary(config: Config, model):
    summarys = config.Summary.split("_")
    if len(summarys) not in [5, 6]:
        raise ValueError(f"Summary in config has wrong number of descriptions. Expect: `5 or 6`, got: `{len(summarys)}`.")
    bits = int(summarys[0][:-4])
    if bits != model.Bits:
        raise ValueError(f"Bits in summary not equals to actual model bits. Expected: `{model.Bits}`, got: `{bits}`.")
    modelType = summarys[1]
    if modelType != str(model.Type):
        raise ValueError(f"Model type in summary not equals to actual model type. Expected: `{str(model.Type)}`, got: `{modelType}`.")



def nop(*_, **__):
    pass


def totalParameters(model: nn.Module) -> str:
    allParams = sum(p.numel() for p in model.parameters())
    unit, suffix = filesize.pick_unit_and_suffix(allParams, ["", "k", "M", "B"], 1000)
    return f"{(allParams / unit):.4f}{suffix}"

def concatOfFiles(paths: List[StrPath], block=io.DEFAULT_BUFFER_SIZE):
    streams = [io.open(path, mode='rb', closefd=True) for path in paths]
    class ChainStream(RawIOBase):
        def __init__(self):
            self.leftover = b''
            self.stream_iter = iter(streams)
            try:
                self.stream = next(self.stream_iter)
            except StopIteration:
                self.stream = None

        def readable(self):
            return True

        def _read_next_chunk(self, max_length):
            # Return 0 or more bytes from the current stream, first returning all
            # leftover bytes. If the stream is closed returns b''
            if self.leftover:
                return self.leftover
            elif self.stream is not None:
                return self.stream.read(max_length)
            else:
                return b''

        def readinto(self, b):
            buffer_length = len(b)
            chunk = self._read_next_chunk(buffer_length)
            while len(chunk) == 0:
                # move to next stream
                if self.stream is not None:
                    self.stream.close()
                try:
                    self.stream = next(self.stream_iter)
                    chunk = self._read_next_chunk(buffer_length)
                except StopIteration:
                    # No more streams to chain together
                    self.stream = None
                    return 0  # indicate EOF
            output, self.leftover = chunk[:buffer_length], chunk[buffer_length:]
            b[:len(output)] = output
            return len(output)

    return io.BufferedReader(ChainStream(), buffer_size=block)

def hashOfStream(path: IOBase):
    sha256 = hashlib.sha256()

    now = 0
    while chunk := path.read(65536):
        # Reading is buffered, so we can read smaller chunks.
        sha256.update(chunk)

    hashResult = sha256.hexdigest()
    return hashResult


def hashOfFile(path: StrPath, progress: Optional[Progress] = None):
    sha256 = hashlib.sha256()

    fileSize = os.path.getsize(path)

    if progress is not None:
        task = progress.add_task(f"[ Hash ]", total=fileSize, progress="0.00%", suffix="")

    now = 0

    with open(path, 'rb') as fp:
        while chunk := fp.read(65536):
            # Reading is buffered, so we can read smaller chunks.
            sha256.update(chunk)
            now += 65536
            if progress is not None:
                progress.update(task, advance=65536, progress=f"{now / fileSize * 100 :.2f}%")

    if progress is not None:
        progress.remove_task(task)

    hashResult = sha256.hexdigest()
    return hashResult

def versionCheck(versionStr: str):
    version = StrictVersion(versionStr)
    builtInVersion = StrictVersion(modfire.__version__)

    if builtInVersion < version:
        raise ValueError(f"Version too new. Given {version}, but I'm {builtInVersion} now.")

    major, minor, revision = version.version

    bMajor, bMinor, bRev = builtInVersion.version

    if major != bMajor:
        raise ValueError(f"Major version mismatch. Given {version}, but I'm {builtInVersion} now.")

    if minor != bMinor:
        warnings.warn(f"Minor version mismatch. Given {version}, but I'm {builtInVersion} now.")
    return True

def getRichProgress(disable: bool = False) -> RichProgress:
    return RichProgress("[i blue]{task.description}[/][b magenta]{task.fields[progress]}", TimeElapsedColumn(), BarColumn(None), TimeRemainingColumn(), "{task.fields[suffix]}", refresh_per_second=6, transient=True, disable=disable, expand=True)


class SafeTerminate(abc.ABC):
    def __init__(self, logger: Optional[LoggerBase]= None):
        self.logger = logger or logging
        signal.signal(signal.SIGTERM, self._terminatedHandler)

    def _kill(self):
        sleep(Consts.TimeOut)
        self.logger.critical("Timeout exceeds, killed.")
        signal.raise_signal(signal.SIGKILL)

    # Handle SIGTERM when main process is terminated.
    # Save necessary info.
    def _terminatedHandler(self, signum, frame):
        killer = threading.Thread(target=self._kill, daemon=True)
        killer.start()
        self.onTerminate(signum, frame)

        self.logger.critical("QUIT.")
        # reset to default SIGTERM handler
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.raise_signal(signal.SIGTERM)


    @abc.abstractmethod
    def onTerminate(self, signum, frame):
        raise NotImplementedError


class ValueBase(Restorable):
    _value: float
    def __init__(self, initValue: float):
        super().__init__()
        self._epoch = 0
        self._initValue = initValue
        self._value = self.calc()

    def step(self):
        self._epoch += 1
        self._value = self.calc()

    def calc(self):
        return self._initValue

    @property
    def Value(self) -> float:
        return self._value


class ConcatTensor(nn.Module):
    _buffer: torch.Tensor
    _DEFAULT_INCREASE = 2048
    def __init__(self, dim: int = 0):
        super().__init__()
        self._dim = dim
        self.register_buffer("_buffer", torch.empty([]))
        self._length = -1

    def reset(self):
        self._length = -1
        self._buffer = torch.empty([], device = self._buffer.device)

    def seek(self, idx: int):
        self._length = idx + 1

    @property
    def Value(self) -> torch.Tensor:
        return self._buffer.index_select(self._dim, torch.arange(self._length))

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self._length < 0:
            shape = list(x.shape)
            shape[self._dim] = math.ceil(x.shape[self._dim] / float(self._DEFAULT_INCREASE)) * self._DEFAULT_INCREASE

            self._buffer = torch.empty(shape, dtype=x.dtype, device=self._buffer.device, pin_memory=True)

            # [1, ..., 1, dim, 1, ..., 1]
            idxShape = [x if i == self._dim else 1 for i, x in enumerate(x.shape)]
            # same shape as x
            idx = torch.arange(0, x.shape[self._dim]).view(idxShape).expand_as(x)
            self._buffer.scatter_(self._dim, idx, x)
            # pointer at end of sequence
            self._length = x.shape[self._dim]
        else:
            if x.shape[self._dim] + self._length >= self._buffer.shape[self._dim]:
                # increase size
                shape = list(x.shape)
                shape[self._dim] = math.ceil(x.shape[self._dim] / float(self._DEFAULT_INCREASE)) * self._DEFAULT_INCREASE
                self._buffer = torch.cat([self._buffer.detach().clone(), torch.empty(shape, dtype=x.dtype, device=self._buffer.device, pin_memory=True)], dim=self._dim)

            # [1, ..., 1, dim, 1, ..., 1]
            idxShape = [x if i == self._dim else 1 for i, x in enumerate(x.shape)]
            # same shape as x
            idx = torch.arange(self._length, self._length + x.shape[self._dim]).view(idxShape).expand_as(x)
            self._buffer.scatter_(self._dim, idx, x)
            self._length += x.shape[self._dim]
