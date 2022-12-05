import pathlib

import click
import torch
import torch.hub
from vlutils.utils import DefaultGroup

import modfire



def version(ctx, _, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(r"""
                      _  __ _
  _ __ ___   ___   __| |/ _(_)_ __ ___
 | '_ ` _ \ / _ \ / _` | |_| | '__/ _ \
 | | | | | | (_) | (_| |  _| | | |  __/
 |_| |_| |_|\___/ \__,_|_| |_|_|  \___|

""" + modfire.__version__)
    ctx.exit()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.group(cls=DefaultGroup, context_settings=CONTEXT_SETTINGS)
@click.option("-v", "--version", is_flag=True, callback=version, expose_value=False, is_eager=True, help="Print version info.")
def entryPoint():
    pass



@entryPoint.command(default=True)
def defaultEntry():
    pass


@entryPoint.command()
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-r", "--resume", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1, help="`.ckpt` file path to resume training.")
@click.argument('config', type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1)
def train(debug, quiet, resume, config):
    """Train a model.

Args:

    config (str): Config file (yaml) path. If `-r/--resume` is present but config is still given, then this config will be used to update the resumed training.
    """
    from modfire.train.cli import main
    main(debug, quiet, resume, config)


@entryPoint.command()
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-e", "--export", is_flag=True, help="If True, append the test results as `result.yaml` under the same dir of test config path. Meanwhile, export the final model to this dir which can be used in the main program. Model name is generated automatically.")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("test", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
def validate(debug, quiet, export, path, test):
    """Validate a trained model from `path` by images from `images` dir, and publish a final state_dict to `output` path.

Args:

    path (str): Saved checkpoint path.

    test (str): A TestConfig path. Please check the TestConfig spec for details.
    """
    from modfire.validate.cli import main
    with torch.inference_mode():
        main(debug, quiet, export, path, test)


@entryPoint.command()
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("--root", required=True, type=click.Path(exists=False, file_okay=False, resolve_path=True, path_type=pathlib.Path), nargs=1)
@click.argument("dataset", type=str, required=True, nargs=1)
def dataset(debug: bool, quiet: bool, root: pathlib.Path, dataset: str):
    """Create training set from `images` dir to `output` dir.

Args:

    dataset (str): Dataset key.
    """
    from modfire.dataset.cli import main
    main(debug, quiet, root, dataset)
