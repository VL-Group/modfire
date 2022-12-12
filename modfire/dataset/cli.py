import pathlib
import click
import logging

from vlutils.logger import configLogging

from modfire import Consts
import modfire.dataset


def checkArgs(debug: bool, quiet: bool):
    if quiet:
        return logging.CRITICAL
    if debug:
        return logging.DEBUG
    return logging.INFO

def main(debug: bool, quiet: bool, root: pathlib.Path, dataset: str):
    loggingLevel = checkArgs(debug, quiet)
    logger = configLogging(rootName="root", level=loggingLevel)
    registry = modfire.dataset.DatasetRegistry
    if dataset is None:
        logger.info("Available datasets are:")
        logger.info(registry.summary())
        return
    if registry.get(dataset, logger).prepare(root, logger):
        logger.info("Dataset preparation finished.")
    else:
        raise RuntimeError("Dataset preparation failed.")

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("--root", required=False, type=click.Path(exists=False, file_okay=False, resolve_path=True, path_type=pathlib.Path))
@click.argument("dataset", type=str, required=False, nargs=1)
def entryPoint(debug: bool, quiet: bool, root: pathlib.Path, dataset: str):
    """Create training set from `images` dir to `output` dir.

Args:

    dataset (optional, str): Dataset key. If not supplied, print all available datasets.
    """
    main(debug, quiet, root, dataset)
