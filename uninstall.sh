#!/bin/bash
set -e
set -o pipefail

if ! command -v conda &> /dev/null
then
    exit
fi

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate base
conda env remove -n modfire
conda deactivate
