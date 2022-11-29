#!/bin/bash
set -e
set -o pipefail

if ! command -v conda &> /dev/null
then
    exit
fi

eval "$(conda shell.bash hook)"
conda activate base
conda env remove -n modfire
