import os
import warnings
import click
import pathlib
import logging
import yaml

import torch
from vlutils.logger import configLogging
from vlutils.logger import trackingFunctionCalls

import modfire
from modfire.utils.registry import ModelRegistry, DatasetRegistry
from modfire.config import Config, TestConfig
from modfire.utils import hashOfFile, versionCheck, getRichProgress, checkConfigSummary

from .validator import Validator


def checkArgs(debug: bool, quiet: bool):
    if quiet:
        return logging.CRITICAL
    if debug:
        return logging.DEBUG
    return logging.INFO


def parseSummary(summary: str):
    summarys = summary.split("_")
    comments = summarys[5] if len(summarys) > 5 else None
    return summarys[0], summarys[1], summarys[2], summarys[3], summarys[4], comments

def appendResult(resultPath: pathlib.Path, summary: str, results: dict):
    with open(resultPath, "r") as fp:
        results = yaml.full_load(fp)
    bits, modelType, method, backbone, trainSet, comments = parseSummary(summary)
    if modelType not in results:
        results[modelType] = dict()
    methods = results[modelType]
    if method not in methods:
        methods[method] = dict()
    backbones = methods[method]
    if backbone not in backbones:
        backbones[backbone] = dict()
    allBits = backbones[backbone]
    if str(bits) not in allBits:
        allBits[bits] = dict()
    trainSets = allBits[bits]

    if comments is not None:
        results.update({ "comments": comments })

    trainSets[trainSet] = results

    with open(resultPath, "w") as fp:
        yaml.dump(results, fp)


def main(debug: bool, quiet: bool, export: bool, path: pathlib.Path, test: pathlib.Path):
    loggingLevel = checkArgs(debug, quiet)

    logger = configLogging(None, "root", loggingLevel)

    checkpoint = torch.load(path, "cuda")

    config = Config.deserialize(checkpoint["config"])
    testConfig = TestConfig.deserialize(yaml.full_load(test.read_text()))

    model = trackingFunctionCalls(ModelRegistry.get(config.Model.Key), logger)(**config.Model.Params).cuda().eval()

    if "trainer" in checkpoint:
        modelStateDict = {key[len("module."):]: value for key, value in checkpoint["trainer"]["_model"].items()}
    else:
        modelStateDict = checkpoint["model"]
        if export:
            warnings.warn("I got an already-converted ckpt.")
        if not "version" in checkpoint:
            raise RuntimeError("You are using a too old version of ckpt, since there is no `version` in it.")
        versionCheck(checkpoint["version"])

    model.load_state_dict(modelStateDict)

    checkConfigSummary(config, model)

    validator = Validator(testConfig.NumReturns)

    progress = getRichProgress()

    with progress:
        result, summary = validator.validate(model, DatasetRegistry.get(testConfig.Database.Key)(**testConfig.Database.Params).Database, DatasetRegistry.get(testConfig.QuerySet.Key)(**testConfig.QuerySet.Params).QuerySet, progress)
        logger.info(summary)

    if not export:
        logger.info(f"Skip exporting model and saving result.")
        return

    resultPath = test.parent.joinpath(f"results.yaml")
    logger.info("Append result to %s", resultPath)

    appendResult(resultPath, config.Summary, result)

    finalName = test.parent.joinpath(f"{config.Summary}.modfire")

    torch.save({
        "model": model.state_dict(),
        "test": {
            "config": testConfig.serialize(),
            "results": summary
        },
        "version": modfire.__version__
    }, finalName)

    logger.info(f"Saved at `{finalName}`.")
    logger.info("Add hash to file...")

    with progress:
        hashResult = hashOfFile(finalName, progress)

    newName = f"{finalName.stem}_{hashResult[:8]}{finalName.suffix}"

    os.rename(finalName, finalName.parent.joinpath(newName))

    logger.info("Rename file to %s", newName)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-e", "--export", is_flag=True, help="If True, append the test results as `result.yaml` under the same dir of test config path. Meanwhile, export the final model to this dir which can be used in the main program. Model name is generated automatically.")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("test", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
def entryPoint(debug, quiet, export, path, test):
    """Validate a trained model from `path` by images from `images` dir, and publish a final state_dict to `output` path.

Args:

    path (str): Saved checkpoint path.

    test (str): A TestConfig path. Please check the TestConfig spec for details.
    """
    main(debug, quiet, export, path, test)
