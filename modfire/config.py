from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional

from marshmallow import Schema, fields, post_load, RAISE


class GeneralSchema(Schema):
    class Meta:
        unknown = RAISE
    key = fields.Str(required=True, description="A unique key used to retrieve in registry. For example, given `Lamb` for optimizers, it will check `OptimRegistry` and find the optimizer `apex.optim.FusedLAMB`.")
    params = fields.Dict(required=True, description="Corresponding funcation call parameters. So the whole call is `registry.get(key)(**params)`.")

    @post_load
    def _(self, data, **kwargs):
        return General(**data)

# Ununsed, unless you want ddp training
class GPUSchema(Schema):
    class Meta:
        unknown = RAISE
    gpus = fields.Int(required=True, description="Number of gpus for training. This affects the `world size` of PyTorch DDP.", exclusiveMinimum=0)
    vRam = fields.Int(required=True, description="Minimum VRam required for each gpu. Set it to `-1` to use all gpus.")
    wantsMore = fields.Bool(required=True, description="Set to `true` to use all visible gpus and all VRams and ignore `gpus` and `vRam`.")

    @post_load
    def _(self, data, **kwargs):
        return GPU(**data)

class TrainSchema(Schema):
    class Meta:
        unknown = RAISE
    epoch = fields.Int(required=True, description="Total training epochs.", exclusiveMinimum=0)
    valFreq = fields.Int(required=True, description="Run validation after every `valFreq` epochs.", exclusiveMinimum=0)
    earlyStop = fields.Int(required=True, description="Early stop after how many evaluations.", exclusiveMinimum=0)
    numReturns = fields.Int(required=True, description="Rank list return number of samples.", exclusiveMinimum=0)
    trainSet = fields.Nested(GeneralSchema(), required=True, description="A spec to load images per line for training.")
    database = fields.Nested(GeneralSchema(), required=True, description="A spec to load images per line for evalution database.")
    querySet = fields.Nested(GeneralSchema(), required=True, description="A spec to load images per line for evalution query.")
    trainer = fields.Str(required=False, default="BaseTrainer", description="A key to retrieve from TrainerBuilder, default is `BaseTrainer`.")
    saveDir = fields.Str(required=True, description="A dir path to save model checkpoints, TensorBoard messages and logs.")
    criterion = fields.Nested(GeneralSchema(), required=True, description="Loss function used for training.")
    optim = fields.Nested(GeneralSchema(), required=True, description="Optimizer used for training. As for current we have `Adam` and `Lamb`.")
    schdr = fields.Nested(GeneralSchema(), required=True, description="Learning rate scheduler used for training. As for current we have `ReduceLROnPlateau`, `Exponential`, `MultiStep`, `OneCycle` and all schedulers defined in `modfire.train.lrSchedulers`.")
    gpu = fields.Nested(GPUSchema(), required=True, description="GPU configs for training.")
    hooks = fields.List(fields.Nested(GeneralSchema()), required=False, description="Hooks used for training. Key is used to retrieve hook from `LBHash.train.hooks`.")
    externalLib = fields.List(fields.Str(), required=False, allow_none=True, description="External libraries used for training. All python files in `externalLib` will be imported as modules. In this way, you could extend registries.")

    @post_load
    def _(self, data, **kwargs):
        return Train(**data)

class ConfigSchema(Schema):
    class Meta:
        unknown = RAISE
    model = fields.Nested(GeneralSchema(), required=True, description="Model to use. Avaliable params are e.g. `backbone`, `bits` and `hashMethod`.")
    train = fields.Nested(TrainSchema(), required=True, description="Training configs.")

    @post_load
    def _(self, data, **kwargs):
        return Config(**data)


class TestConfigSchema(Schema):
    class Meta:
        unknown = RAISE
    numReturns = fields.Int(required=True, description="Rank list return number of samples.", exclusiveMinimum=0)
    database = fields.Nested(GeneralSchema(), required=True, description="A spec to load images per line for evalution database.")
    querySet = fields.Nested(GeneralSchema(), required=True, description="A spec to load images per line for evalution query.")

    @post_load
    def _(self, data, **kwargs):
        return TestConfig(**data)


@dataclass
class General:
    key: str
    params: Dict[str, Any]

    @property
    def Key(self) -> str:
        return self.key

    @property
    def Params(self) -> Dict[str, Any]:
        return self.params

@dataclass
class GPU:
    gpus: int
    vRam: int
    wantsMore: bool

    @property
    def GPUs(self) -> int:
        return self.gpus

    @property
    def VRam(self) -> int:
        return self.vRam

    @property
    def WantsMore(self) -> bool:
        return self.wantsMore

@dataclass
class Train:
    epoch: int
    valFreq: int
    trainer: str
    earlyStop: int
    trainSet: General
    database: General
    querySet: General
    saveDir: str
    optim: General
    schdr: General
    criterion: General
    numReturns: int
    gpu: GPU
    hooks: Optional[List[General]] = None
    externalLib: Optional[List[str]] = None

    @property
    def Epoch(self) -> int:
        return self.epoch

    @property
    def ValFreq(self) -> int:
        return self.valFreq

    @property
    def EarlyStop(self) -> int:
        return self.earlyStop

    @property
    def Trainer(self) -> str:
        return self.trainer

    @property
    def NumReturns(self) -> int:
        return self.numReturns

    @property
    def TrainSet(self) -> General:
        return self.trainSet

    @property
    def Database(self) -> General:
        return self.database

    @property
    def QuerySet(self) -> General:
        return self.querySet

    @property
    def SaveDir(self) -> str:
        return self.saveDir

    @property
    def Optim(self) -> General:
        return self.optim

    @property
    def Schdr(self) -> General:
        return self.schdr

    @property
    def Criterion(self) -> General:
        return self.criterion

    @property
    def GPU(self) -> GPU:
        return self.gpu

    @property
    def Hooks(self) -> List[General]:
        if self.hooks is None:
            return list()
        return self.hooks

    @property
    def ExternalLib(self) -> List[str]:
        if self.externalLib is None:
            return list()
        return self.externalLib


@dataclass
class Config:
    model: General
    train: Train

    @property
    def Model(self) -> General:
        return self.model

    @property
    def Train(self) -> Train:
        return self.train

    def serialize(self) -> dict:
        return ConfigSchema().dump(self)

    @staticmethod
    def deserialize(data: dict) -> "Config":
        # patch for the speical "$schema" key in json
        data = { key: value for key, value in data.items() if "$" not in key }
        return ConfigSchema().load(data)


@dataclass
class TestConfig:
    querySet: General
    database: General
    numReturns: int

    @property
    def QuerySet(self) -> General:
        return self.querySet

    @property
    def Database(self) -> General:
        return self.database

    @property
    def NumReturns(self) -> int:
        return self.numReturns

    def serialize(self) -> dict:
        return TestConfigSchema().dump(self)

    @staticmethod
    def deserialize(data: dict) -> "TestConfig":
        # patch for the speical "$schema" key in json
        data = { key: value for key, value in data.items() if "$" not in key }
        return TestConfigSchema().load(data)
