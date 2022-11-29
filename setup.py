from setuptools import setup
import os

setupArgs = dict()

if "ADD_ENTRY" in os.environ:
    console_scripts = [
        "modfire = modfire.cli:entryPoint",
        "modfire-train = modfire.train.cli:entryPoint",
        "modfire-validate = modfire.validate.cli:entryPoint"
    ]
    setupArgs.update({
        "entry_points": {
            'console_scripts': console_scripts
        }
    })


setup(**setupArgs)
