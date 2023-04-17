import math
from copy import deepcopy
import os
import shutil
from typing import Callable, Tuple, Dict, Union, Any
import gc
import pathlib
import hashlib
import importlib.util
import sys
import numbers

import torch
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from vlutils.base import Registry
from vlutils.base.freqHook import ChainHook
from vlutils.saver import Saver
from vlutils.logger import trackingFunctionCalls
from vlutils.base import Restorable
from vlutils.runtime import relativePath
from vlutils.config import summary

import modfire.utils.registry
from modfire.utils.registry import OptimRegistry, SchdrRegistry, CriterionRegistry, ModelRegistry, DatasetRegistry, DataPipeRegistry
from modfire import Consts
from modfire.config import Config
from modfire.train.hooks import getAllHooks
from modfire.validate import Validator, metrics
from modfire.utils import totalParameters, StrPath, getRichProgress, SafeTerminate
from modfire.dataset import QuerySplit, Database, TrainSplit

from .hooks import EpochFrequencyHook, checkHook, splitHooks
from .utils import EMATracker, PrettyStep, getSaver, setWeightDecay


class TrainerBuilder(modfire.utils.registry.Registry[Callable[..., "PalTrainer"]]):
    pass


def _p(attr):
    def _decorate(self):
        msg = "[%s] " + attr
        return msg, self.PrettyStep
    return property(_decorate)

class LoggingTemplate:
    def __init__(self):
        self._step = 0
        self._prettyStep = PrettyStep()

    @property
    def PrettyStep(self):
        return self._prettyStep(self._step)


    ########### Templates ###########
    _LOG_InitSummary   = _p("Here is the whole config during this run: \r\n%s")
    _LOG_Bye           = _p("Bye.")
    _LOG_FindCkpt      = _p("Found ckpt to resume at %s")
    _LOG_ResumeAtEpoch = _p("Resume training at %d epochs.")
    _LOG_StartTraining = _p("Start training.")
    _LOG_RevealTensorboard = _p("See you at `%s`")
    _LOG_EarlyStop = _p("Early stopped at epoch %4d.")
    _LOG_SummaryInfoNomAP = _p("Total epoches: %d, total steps: %s, best mAP: N/A.")
    _LOG_SummaryInfo = _p("Total epoches: %d, total steps: %s, best mAP: %.2f%%.")
    _LOG_SummarySaveCkpt = _p("Model saved to %s`.")


    _DEBUG_InitRank           = _p("<%s> is located at rank `%d`")
    _DEBUG_Inited             = _p("<%s> created.")
    _DEBUG_RestoreStateStart  = _p("Restored state dict from `%s`")
    _DEBUG_RestoreStateFinish = _p("Restore network parameters finished.")
    _DEBUG_OptimizerReset     = _p("Optimizer reset.")
    _DEBUG_LRSchdrReset       = _p("LR scheduler reset.")
    _DEBUG_RegistrySummary    = _p("Summary of %s: \r\n%s")
    _DEBUG_CreateDatasetStart = _p("Create `config.Train.TrainSet` (\"%s\") with training pipeline: `%s`.")
    _DEBUG_CreateQuerysetStart = _p("Create `config.Train.QuerySet` (\"%s\") with evaluation pipeline: `%s`.")
    _DEBUG_CreateDatabaseStart = _p("Create `config.Train.Database` (\"%s\") with evaluation pipeline: `%s`.")
    _DEBUG_CreateDatasetFinish = _p("Training dataset \r\n\t%s \r\nmounted.")
    _DEBUG_CreateDatasetMainFinish = _p("Train set \r\n\t%s, \r\nquery set \r\n\t%s and \r\ndatabase \r\n\t%s \r\nmounted.")
    _DEBUG_CreateModelStart = _p("Creating model...")
    _DEBUG_CreateModelFinish = _p("Model created. #Params: %s.")
    _DEBUG_CreateCriterionStart = _p("Creating criterion...")
    _DEBUG_CreateCriterionFinish = _p("criterion created. #Params: %s.")
    _DEBUG_CreateOptimizerStart = _p("Creating optimizer...")
    _DEBUG_CreateOptimizerFinish = _p("Optimizer created.")
    _DEBUG_CreateSchdrStart = _p("Creating LR scheduler...")
    _DEBUG_CreateSchdrFinish = _p("LR scheduler created.")
    _DEBUG_CallBeforRunStart = _p("Call `_beforeRun()`.")
    _DEBUG_CallBeforRunFinish = _p("End call `_beforeRun()`.")
    _DEBUG_CallAfterRunStart = _p("Call `_afterRun()`.")
    _DEBUG_CallAfterRunFinish = _p("End call `_afterRun()`.")
    _DEBUG_CallEpochBeginStart = _p("Epoch %4d started.")
    _DEBUG_CallEpochBeginFinish = _p("End call `_epochStart()`.")
    _DEBUG_CallEpochEndStart = _p("Epoch %4d finished.")
    _DEBUG_CallEpochEndFinish = _p("End call `_epochFinish()`.")
    _DEBUG_TrainLoopFinish = _p("Training loop finished.")
    _DEBUG_ShowSchdrUpdate = _p("Lr is set to %.2e.")
    _DEBUG_ShowTemperatureUpdate = _p("Temperature is set to %.2e.")
    _DEBUG_Validation = _p("Start validation at epoch %4d.")
    _DEBUG_FindOneEarlyStop = _p("Performance not improved for %d / %d epochs.")
    _DEBUG_EndValidation = _p("End validation at epoch %4d.")

    _CRITICAL_ProcessInterrupt = _p("Main process was interrupted, try to save necessary info.")
    _CRITICAL_TimeoutInfo = _p("This post-process will be killed after %d secs if stuck.")
    _CRITICAL_ShutdownMessage = _p("Find the last checkpoint at `%s`")


class PalTrainer(Restorable, LoggingTemplate):
    def __init__(self, config: Config, loggingLevel: int):
        Restorable.__init__(self)
        LoggingTemplate.__init__(self)

        self._epoch = 0

        self.rank = dist.get_rank()
        self.worldSize = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.config = config
        self.saver = getSaver(self.config.Train.SaveDir, saveName="saved.ckpt", config=config.serialize(), loggerName="root", reserve=False, loggingLevel=loggingLevel, disable=self.rank != 0)

        self.saver.info(*self._LOG_InitSummary, summary(config.serialize()))

        self.saver.debug(*self._DEBUG_InitRank, self.__class__.__name__, self.rank)

        # # Used for self.PrettyStep
        # self.lastFormatted = -1
        self._preRegistration()

        self._model = self._createModel()
        self._criterion = self._createCriterion()
        self._optimizer, self.optimFn = self._createOptimizer(self._model, self._criterion)
        self._scheduler, self.schdrFn = self._createScheduler(self._optimizer)

        self.earlyStopFlag = torch.tensor([False]).to(self.rank)

        self.saver.debug(*self._DEBUG_Inited, self.__class__.__name__)


    def save(self, path = None):
        self.saver.save(path, trainer=self, config=self.config.serialize())

    def done(self):
        self.saver.debug(summary(self.config.serialize()))
        self.saver.info(*self._LOG_Bye)

    def resume(self, path):
        self.saver.info(*self._LOG_FindCkpt, path)
        self.restoreStates(path)
        self.saver.info(*self._LOG_ResumeAtEpoch, self._epoch)

    def restoreStates(self, path: StrPath):
        self.saver.debug(*self._DEBUG_RestoreStateStart, path)
        self.saver.load(path, "cpu", logger=self.saver, trainer=self)
        self.saver.debug(*self._DEBUG_RestoreStateFinish)
        self.resetOptimizer()
        self.resetScheduler(*self._scheduler.last_epoch)

    def resetOptimizer(self):
        del self._optimizer
        self._optimizer = self.optimFn(self._model.parameters(), **self.config.Train.Optim.Params)

        for group in self._optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.saver.debug(self._DEBUG_OptimizerReset)

    def resetScheduler(self, lastEpoch=-1):
        del self._scheduler
        self._scheduler = self.schdrFn(self._optimizer, last_epoch=lastEpoch, **self.config.Train.Schdr.Params)
        self.saver.debug(self._DEBUG_LRSchdrReset)

    def train(self):
        beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook = self._createHooks(self.config, self.saver, self._model, self._criterion)

        scaler = GradScaler()

        datasets = self._createDatasets()

        # The DistributedReadingService is too slow since it use only one worker per node.
        # NOTE: cancel the comment once if the above issue is fixed.
        # with DataLoader(datasets["trainSet"].DataPipe, reading_service=DistributedReadingService()) as trainLoader:
        with DataLoader2(datasets["trainSet"].DataPipe, reading_service=MultiProcessingReadingService(num_workers=min(int(math.sqrt(datasets["trainSet"].BatchSize)), 16), pin_memory=True, persistent_workers=True)) as trainLoader:

            self._beforeRun(beforeRunHook, **datasets)

            batchesOneEpoch = math.ceil(len(datasets["trainSet"]) / (datasets["trainSet"].BatchSize * self.worldSize))
            totalBatches = batchesOneEpoch * self.config.Train.Epoch

            self._model.train()

            # A forever dataLoader
            for targets, images in trainLoader:
                if self._step % batchesOneEpoch == 0:
                    self._epochStart(epochStartHook, **datasets)

                with autocast():

                    # Main loop
                    # Any dict used as args for model, criterion
                    otherArgs = self._stepStart(stepStartHook, inputs=(targets, images))

                    # A dict as keyword arguments for criterion
                    outputs = self._model(images.to(self.rank, non_blocking=True, memory_format=torch.channels_last), **otherArgs)

                    # loss: A scalar, stats: A dict as keyword arguments for logging
                    loss, stats = self._criterion(**outputs, y=targets.to(self.rank, non_blocking=True), **otherArgs)

                scaler.scale(loss).backward()
                scaler.step(self._optimizer)
                scaler.update()
                self._optimizer.zero_grad()


                self._stepFinish(stepFinishHook, loss=loss, stats=stats, outputs=outputs, **otherArgs)
                if self._step % batchesOneEpoch == 0:
                    try:
                        self._epochFinish(epochFinishHook, **datasets)
                    except StopIteration:
                        break
                if self._step > totalBatches:
                    break

        self._afterRun(afterRunHook, **datasets)


    def _preRegistration(self):
        otherPythonFiles = self.config.Train.ExternalLib
        for pyFile in otherPythonFiles:
            filePath = pathlib.Path(pyFile).absolute()
            # md5 of abs file path as module name
            moduleName = hashlib.md5(str(filePath).encode()).hexdigest()
            spec = importlib.util.spec_from_file_location(moduleName, pyFile)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[moduleName] = module
            spec.loader.exec_module(module)

        for reg in modfire.utils.registry.__all__:
            registry = getattr(modfire.utils.registry, reg)
            if issubclass(registry, Registry):
                self.saver.debug(*self._DEBUG_RegistrySummary, registry, registry.summary())

    @staticmethod
    def _createHooks(config: Config, saver: Saver, model, criterion):
        allHooks = getAllHooks(config.Train.Hooks)

        modelHooks = splitHooks(*[m.module if isinstance(m, DistributedDataParallel) else m for m in [model, criterion]])

        # allHooks = dict()
        for key in modelHooks.keys():
            allHooks[str(key)] = ChainHook(allHooks[str(key)], modelHooks[key])


        beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook = allHooks["beforeRunHook"], allHooks["afterRunHook"], allHooks["stepStartHook"], allHooks["stepFinishHook"], allHooks["epochStartHook"], allHooks["epochFinishHook"]


        beforeRunHook = checkHook(beforeRunHook, "BeforeRunHook", saver)
        afterRunHook = checkHook(afterRunHook, "AfterRunHook", saver)
        stepStartHook = checkHook(stepStartHook, "StepStartHook", saver)
        stepFinishHook = checkHook(stepFinishHook, "StepFinishHook", saver)
        epochStartHook = checkHook(epochStartHook, "EpochStartHook", saver)
        epochFinishHook = checkHook(epochFinishHook, "EpochFinishHook", saver)
        return beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook

    def _createDatasets(self) -> Dict[str, Union[TrainSplit, QuerySplit, Database]]:
        self.saver.debug(*self._DEBUG_CreateDatasetStart, self.config.Train.TrainSet.Key, self.config.Train.TrainSet.Pipeline.Key or "default")
        try:
            trainPipeline = trackingFunctionCalls(DataPipeRegistry.get(self.config.Train.TrainSet.Pipeline.Key), self.saver)(**self.config.Train.TrainSet.Pipeline.Params)
        except KeyError:
            trainPipeline = None
        trainSet = trackingFunctionCalls(DatasetRegistry.get(self.config.Train.TrainSet.Key), self.saver)(**self.config.Train.TrainSet.Params, pipeline=trainPipeline)
        self.saver.debug(*self._DEBUG_CreateDatasetFinish, trainSet)
        return {
            "trainSet": trainSet.Split
        }

    def _createModel(self) -> DistributedDataParallel:
        self.saver.debug(*self._DEBUG_CreateModelStart)
        modelFn = trackingFunctionCalls(ModelRegistry.get(self.config.Model.Key), self.saver)

        model = modelFn(**self.config.Model.Params)

        # EMA model for evaluation
        # deepcopy can't handle faiss objects. reject.
        # adjust = worldSize * config.Train.TrainSet.Params["batchSize"] * config.Train.ModelEMASteps / config.Train.Epoch
        # alpha = 1.0 - config.Train.ModelEMADecay
        # alpha = min(1.0, alpha * adjust)
        # modelEMA = ExponentialMovingAverage(model, device=rank, decay=1.0 - alpha)
        # EMA model for evaluation

        model = DistributedDataParallel(model.to(memory_format=torch.channels_last).to(self.rank), device_ids=[self.rank], output_device=self.rank, find_unused_parameters=False)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.saver.debug(*self._DEBUG_CreateModelFinish, totalParameters(model))
        return model

    def _createCriterion(self) -> DistributedDataParallel:
        self.saver.debug(*self._DEBUG_CreateCriterionStart)
        criterionFn = trackingFunctionCalls(CriterionRegistry.get(self.config.Train.Criterion.Key), self.saver)
        criterion = criterionFn(**self.config.Train.Criterion.Params).to(self.rank)
        if any((p.requires_grad for p in criterion.parameters())):
            criterion = DistributedDataParallel(criterion, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=False)
            criterion = torch.nn.SyncBatchNorm.convert_sync_batchnorm(criterion)
        self.saver.debug(*self._DEBUG_CreateCriterionFinish, totalParameters(criterion))
        return criterion

    def _createOptimizer(self, model: DistributedDataParallel, criterion: DistributedDataParallel) -> Tuple[torch.optim.Optimizer, Callable[..., torch.optim.Optimizer]]:
        self.saver.debug(*self._DEBUG_CreateOptimizerStart)
        if "lr" in self.config.Train.Optim.Params and "batchSize" in self.config.Train.TrainSet.Params:
            batchSize = self.config.Train.TrainSet.Params["batchSize"] * self.worldSize
            exponent = math.log2(batchSize)
            scale = 3 - exponent / 2
            optimCfg = deepcopy(self.config.Train.Optim)
            optimCfg.Params["lr"] /= (2 ** scale)
        else:
            optimCfg = self.config.Train.Optim
        optimFn = trackingFunctionCalls(OptimRegistry.get(optimCfg.Key), self.saver)

        # remove weight_decay in any norm layers
        paramGroup = setWeightDecay(model, optimCfg.Params["weight_decay"], 0.0) +\
                        setWeightDecay(criterion, optimCfg.Params["weight_decay"], 0.0)

        optimizer = optimFn(paramGroup, **optimCfg.Params)
        self.saver.debug(*self._DEBUG_CreateOptimizerFinish)
        return optimizer, optimFn

    def _createScheduler(self, optimizer: torch.optim.Optimizer) -> Tuple[torch.optim.lr_scheduler._LRScheduler, Callable[..., torch.optim.lr_scheduler._LRScheduler]]:
        self.saver.debug(*self._DEBUG_CreateSchdrStart)
        schdrFn = trackingFunctionCalls(SchdrRegistry.get(self.config.Train.Schdr.Key), self.saver)
        scheduler = schdrFn(optimizer, **self.config.Train.Schdr.Params)
        self.saver.debug(*self._DEBUG_CreateSchdrFinish)
        return scheduler, schdrFn

    def _beforeRun(self, hook, *args, **kwArgs):
        self.saver.debug(*self._DEBUG_CallBeforRunStart)
        self.saver.info(*self._LOG_StartTraining)

        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)

        self.saver.info(*self._LOG_RevealTensorboard, self.saver.TensorboardURL)
        self.saver.debug(*self._DEBUG_CallBeforRunFinish)

    def _afterRun(self, hook, *args, **kwArgs):
        self.saver.debug(*self._DEBUG_CallAfterRunStart)
        self.saver.debug(*self._DEBUG_TrainLoopFinish)
        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)
        self.saver.debug(*self._DEBUG_CallAfterRunFinish)

    def _stepStart(self, hook, *args, **kwArgs) -> Dict[str, Any]:
        return hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs) or dict()

    def _stepFinish(self, hook, *args, loss, **kwArgs):
        self._step += 1

        # Update AveragedModel
        # if self._step % self.config.Train.ModelEMASteps == 0:
        #     self._modelEMA.update_parameters(self._model)
        #     if "warmupEpochs" in self.config.Train.Schdr.Params and self._epoch < self.config.Train.Schdr.Params["warmupEpochs"]:
        #         # Reset ema buffer to keep copying weights during warmup period
        #         self._modelEMA.n_averaged.fill_(0)

        hook(self._step, self._epoch, self, *args, logger=self.saver, loss=loss, **kwArgs)

    def _epochStart(self, hook, *args, **kwArgs):
        self.saver.debug(*self._DEBUG_CallEpochBeginStart, self._epoch + 1)

        gc.collect()
        gc.collect()
        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)

        self.saver.debug(*self._DEBUG_CallEpochBeginFinish)

    def _epochFinish(self, hook, *args, **kwArgs):
        self._epoch += 1

        self.saver.debug(*self._DEBUG_CallEpochEndStart, self._epoch)


        dist.broadcast(self.earlyStopFlag, 0)
        if self.earlyStopFlag:
            self.saver.info(*self._LOG_EarlyStop, self._epoch)
            raise StopIteration

        self._scheduler.step()
        self._model.module.step()
        self.saver.debug(*self._DEBUG_ShowSchdrUpdate, self._scheduler.get_last_lr()[0])
        self.saver.debug(*self._DEBUG_ShowTemperatureUpdate, self._model.module.Temperature)

        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)
        self.saver.debug(*self._DEBUG_CallEpochEndFinish)



class MainTrainer(PalTrainer, SafeTerminate):
    def __init__(self, config: Config, loggingLevel: int):
        PalTrainer.__init__(self, config, loggingLevel)
        SafeTerminate.__init__(self, self.saver)
        # Running depedencies
        self.progress = getRichProgress().__enter__()
        self.epochBar = self.progress.add_task("[----/----]", start=False, progress="", suffix=Consts.CDot * 10)
        self.trainingBar = self.progress.add_task("", start=False, progress="[----/----]", suffix=Consts.CDot * 10)

        self.validator = Validator(self.config.Train.NumReturns)

        self.diffTracker = EMATracker((), 0.99).cuda()

        # Logging and saving
        self.bestmAP = -1
        self.earlyStopCount = 0

    def onTerminate(self, signum, frame):
        self.saver.critical(*self._CRITICAL_ProcessInterrupt)
        self.saver.critical(*self._CRITICAL_TimeoutInfo, Consts.TimeOut)
        self.progress.__exit__(None, None, None)
        self.save(os.path.join(self.saver.SaveDir, "last.ckpt"))
        self.saver.critical(*self._CRITICAL_ShutdownMessage, relativePath(os.path.join(self.saver.SaveDir, "last.ckpt")))
        self.summary()

    def summary(self):
        if self.bestmAP < 0:
            self.saver.info(*self._LOG_SummaryInfoNomAP, self._epoch, self._step)
        else:
            self.saver.info(*self._LOG_SummaryInfo, self._epoch, self._step, self.bestmAP * 100)
        self.saver.info(*self._LOG_SummarySaveCkpt, relativePath(os.path.join(self.saver.SaveDir, "[ONE_OF_A].ckpt")))

    def _beforeRun(self, hook, *args, **kwArgs):
        self.progress.start_task(self.trainingBar)
        self.progress.start_task(self.epochBar)
        super()._beforeRun(hook, *args, **kwArgs)

    def _afterRun(self, hook, *args, **kwArgs):
        super()._afterRun(hook, *args, **kwArgs)
        self.progress.__exit__(None, None, None)
        self.summary()

    def _stepFinish(self, hook, *args, loss, stats, **kwArgs):
        super()._stepFinish(hook, *args, loss=loss, **kwArgs)

        moment = self.diffTracker(loss)

        task = self.progress.get_task(self.trainingBar)
        self.progress.update(self.trainingBar, advance=1, progress=f"[{task.completed + 1:4d}/{task.total:4d}]", suffix=f"L = [b green]{moment:2.2f}[/]")
        self.progress.update(self.epochBar, advance=1)

        if self._step % 10 != 0:
            return

        for key, value in stats.items():
            if isinstance(value, numbers.Number):
                self.saver.add_scalar(f"Stat/{key}", value, global_step=self._step)
            else:
                if value.numel() == 1:
                    self.saver.add_scalar(f"Stat/{key}", value, global_step=self._step)
                elif len(value.shape) == 4:
                    self.saver.add_images(f"Stat/{key}", value, global_step=self._step)
                elif len(value.shape) == 3:
                    self.saver.add_image(f"Stat/{key}", value, global_step=self._step)
                else:
                    self.saver.add_histogram(f"Stat/{key}", value, global_step=self._step)
        self.saver.add_scalar("Stat/Loss", loss, global_step=self._step)
        self.saver.add_scalar("Stat/Lr", self._scheduler.get_last_lr()[0], global_step=self._step)

    def _epochStart(self, hook, *args, trainSet: TrainSplit, **kwArgs):
        totalBatches = math.ceil(len(trainSet) / (trainSet.BatchSize * self.worldSize))
        self.progress.update(self.trainingBar, total=totalBatches)
        self.progress.update(self.epochBar, total=self.config.Train.Epoch * totalBatches, completed=self._step, description=f"[{self._epoch + 1:4d}/{self.config.Train.Epoch:4d}]")

        self.progress.reset(self.trainingBar)
        super()._epochStart(hook, *args, trainSet=trainSet, **kwArgs)

    def _createHooks(self, config: Config, saver: Saver, model, criterion):
        beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook = super()._createHooks(config, saver, model, criterion)

        saver.debug("Add additional hooks in `MainTrainer`.")

        epochFinishHook = checkHook(ChainHook(
            EpochFrequencyHook(
                (1, self.log), logger=saver
            ),
            EpochFrequencyHook(
                (config.Train.ValFreq, self.validate), logger=saver
            ), epochFinishHook), "EpochFinishHook", saver)

        beforeRunHook = checkHook(ChainHook(
            self.validate
            , beforeRunHook), "BeforeRunHook", saver)

        return beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook

    def _createDatasets(self) -> Dict[str, Union[TrainSplit, QuerySplit, Database]]:
        self.saver.debug(*self._DEBUG_CreateDatasetStart, self.config.Train.TrainSet.Key, self.config.Train.TrainSet.Pipeline.Key or "default")
        try:
            trainPipeline = trackingFunctionCalls(DataPipeRegistry.get(self.config.Train.TrainSet.Pipeline.Key), self.saver)(**self.config.Train.TrainSet.Pipeline.Params)
        except KeyError:
            trainPipeline = None
        trainSet = trackingFunctionCalls(DatasetRegistry.get(self.config.Train.TrainSet.Key), self.saver)(**self.config.Train.TrainSet.Params, pipeline=trainPipeline)


        self.saver.debug(*self._DEBUG_CreateQuerysetStart, self.config.Train.QuerySet.Key, self.config.Train.QuerySet.Pipeline.Key or "default")
        try:
            queryPipeline = trackingFunctionCalls(DataPipeRegistry.get(self.config.Train.QuerySet.Pipeline.Key), self.saver)(**self.config.Train.QuerySet.Pipeline.Params)
        except KeyError:
            queryPipeline = None
        querySet = trackingFunctionCalls(DatasetRegistry.get(self.config.Train.QuerySet.Key), self.saver)(**self.config.Train.QuerySet.Params, pipeline=queryPipeline)


        self.saver.debug(*self._DEBUG_CreateDatabaseStart, self.config.Train.Database.Key, self.config.Train.Database.Key or "default")
        try:
            databasePipeline = trackingFunctionCalls(DataPipeRegistry.get(self.config.Train.Database.Pipeline.Key), self.saver)(**self.config.Train.Database.Pipeline.Params)
        except KeyError:
            databasePipeline = None
        database = trackingFunctionCalls(DatasetRegistry.get(self.config.Train.Database.Key), self.saver)(**self.config.Train.Database.Params, pipeline=databasePipeline)



        self.saver.debug(*self._DEBUG_CreateDatasetMainFinish, trainSet, database, querySet)
        return {
            "trainSet": trainSet.Split,
            "database": database.Split,
            "querySet": querySet.Split
        }

    def log(self, *_, **__):
        self.saver.add_scalar("Stat/Epoch", self._epoch, self._step)
        # self.saver.add_images("Train/Raw", tensorToImage(images), global_step=self._step)

    def validate(self, *_, database: Database, querySet: QuerySplit, **__):
        torch.cuda.empty_cache()

        self.saver.debug(*self._DEBUG_Validation, self._epoch)

        results, summary = self.validator.validate(self._model.module, database, querySet, self.progress)

        for metricModule in metrics.__all__:
            if metricModule != "Visualization":
                # [mAP, Precision, Recall]
                self.saver.add_scalar(f"Eval/{metricModule}@{self.validator.numReturns}", results[metricModule], global_step=self._step)
        # self.saver.add_images(f"Eval/Visualization", results["Visualization"], global_step=self._step)

        self.save()

        self.saver.info("%s", summary)

        mAP = results["mAP"]
        if mAP > self.bestmAP:
            self.bestmAP = mAP
            self.progress.update(self.epochBar, suffix=f"H = [b red]{self.bestmAP * 100:2.2f}[/]%")
            shutil.copy2(self.saver.SavePath, os.path.join(self.saver.SaveDir, "best.ckpt"))
            self.earlyStopCount = 0
        else:
            self.earlyStopCount += 1
            self.saver.debug(*self._DEBUG_FindOneEarlyStop, self.earlyStopCount, self.config.Train.EarlyStop)
            if self.earlyStopCount >= self.config.Train.EarlyStop:
                self.earlyStop()

        self.saver.debug(*self._DEBUG_EndValidation, self._epoch)
        self._model.train()

    def earlyStop(self):
        self.earlyStopFlag.data.copy_(torch.tensor([True]))


@TrainerBuilder.register("BaseTrainer")
def getTrainer(rank: int, config: Config, loggingLevel: int):
    if rank == 0:
        return MainTrainer(config, loggingLevel)
    return PalTrainer(config, loggingLevel)
