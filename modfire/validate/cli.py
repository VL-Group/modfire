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
from modfire.utils import hashOfFile, versionCheck, getRichProgress

from .validator import Validator


def checkArgs(debug: bool, quiet: bool):
    if quiet:
        return logging.CRITICAL
    if debug:
        return logging.DEBUG
    return logging.INFO


def main(debug: bool, quiet: bool, export: pathlib.Path, path: pathlib.Path, test: pathlib.Path):
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
        if export is not None:
            warnings.warn("I got an already-converted ckpt.")
        if not "version" in checkpoint:
            raise RuntimeError("You are using a too old version of ckpt, since there is no `version` in it.")
        versionCheck(checkpoint["version"])

    model.load_state_dict(modelStateDict)

    validator = Validator(testConfig.NumReturns)

    progress = getRichProgress()

    with progress:
        _, summary = validator.validate(model, DatasetRegistry.get(testConfig.Database.Key)(**testConfig.Database.Params).Database, DatasetRegistry.get(testConfig.QuerySet.Key)(**testConfig.QuerySet.Params).QuerySet, progress)
        logger.info(summary)

    if export is None:
        logger.info(f"Skip saving model.")
        return

    finalName = export.joinpath(f"{config.Model.Key}_{model.Type}.modfire")

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
@click.option("-e", "--export", type=click.Path(exists=False, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, help="Dir to export the final model that is compatible with main program. Model name is generated automatically.")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("test", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
def entryPoint(debug, quiet, export, path, test):
    """Validate a trained model from `path` by images from `images` dir, and publish a final state_dict to `output` path.

Args:

    path (str): Saved checkpoint path.

    test (str): A TestConfig path. Please check the TestConfig spec for details.
    """
    main(debug, quiet, export, path, test)
