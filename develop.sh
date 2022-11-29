#!/bin/bash
set -e
set -o pipefail


echo "Start installation"

if ! command -v conda &> /dev/null
then
    echo "conda could not be found, please ensure you've installed conda and place it in PATH."
    exit
fi


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    conda create -y -n modfire "python<3.9" "torchdata<1" pytorch-cuda=11.7 "faiss-cpu<1.8" "torchvision<1" "pytorch>=1.13,<2" -c pytorch -c nvidia
elif [[ "$OSTYPE" == "darwin"* ]]; then
    conda create -y -n modfire "python<3.9" "torchdata<1" "faiss-cpu<1.8" "pytorch>=1.13,<2" "torchvision<1" -c pytorch
else
    conda create -y -n modfire "python<3.9" "torchdata<1" pytorch-cuda=11.7 "faiss-cpu<1.8" "torchvision<1" "pytorch>=1.13,<2" -c pytorch -c nvidia
fi

eval "$(conda shell.bash hook)"

conda activate modfire


if [[ "$CONDA_DEFAULT_ENV" != "modfire" ]]; then
    echo "Can't activate conda env modfire, exit."
    exit
fi

conda install -y -n modfire "tensorboard<3" "rich<11" "python-lmdb<2" "pyyaml<7" "marshmallow<4" "click<9" "msgpack-python<2" "scipy<2" packaging -c conda-forge



ADD_ENTRY=SET pip install -e .


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i "1 s|$| -O|" "$(which modfire)"*
elif [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i "" "1 s|$| -O|" "$(which modfire)"*
else
    sed -i "1 s|$| -O|" "$(which modfire)"*
fi

echo "Installation done!"

echo "If you want to train models, please install NVIDIA/Apex manually."
echo "If you want to use streamlit service, please install streamlit via pip."
