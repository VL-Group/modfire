from ._api import findLastLinear, replaceModule

import logging
import os
import signal
from typing import List, Optional, Union, Any, Dict, Tuple
from pathlib import Path
import hashlib
from time import sleep
import threading
import abc

from vlutils.logger import LoggerBase
from vlutils.custom import RichProgress
from vlutils.saver import StrPath
from vlutils.base import FrequecyHook
import torch
from torch import nn
from rich import filesize
from rich.progress import Progress
from rich.progress import TimeElapsedColumn, BarColumn, TimeRemainingColumn

from modfire import Consts


def nop(*_, **__):
    pass


def totalParameters(model: nn.Module) -> str:
    allParams = sum(p.numel() for p in model.parameters())
    unit, suffix = filesize.pick_unit_and_suffix(allParams, ["", "k", "M", "B"], 1000)
    return f"{(allParams / unit):.4f}{suffix}"


def hashOfFile(path: StrPath, progress: Optional[Progress] = None):
    sha256 = hashlib.sha256()

    fileSize = os.path.getsize(path)

    if progress is not None:
        task = progress.add_task(f"[ Hash ]", total=fileSize, progress="0.00%", suffix="")

    now = 0

    with open(path, 'rb') as fp:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = fp.read(65536)
            if not chunk:
                break
            sha256.update(chunk)
            now += 65536
            if progress is not None:
                progress.update(task, advance=65536, progress=f"{now / fileSize * 100 :.2f}%")

    if progress is not None:
        progress.remove_task(task)

    hashResult = sha256.hexdigest()
    return hashResult

def getRichProgress(disable: bool = False) -> RichProgress:
    return RichProgress("[i blue]{task.description}[/][b magenta]{task.fields[progress]}", TimeElapsedColumn(), BarColumn(None), TimeRemainingColumn(), "{task.fields[suffix]}", refresh_per_second=6, transient=True, disable=disable, expand=True)


class SafeTerminate(abc.ABC):
    def __init__(self, logger: Optional[LoggerBase]= None) -> None:
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

        self.logger.critical("[%s] QUIT.")
        # reset to default SIGTERM handler
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.raise_signal(signal.SIGTERM)


    @abc.abstractmethod
    def onTerminate(self, signum, frame):
        raise NotImplementedError
