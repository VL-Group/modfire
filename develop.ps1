Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSDefaultParameterValues['*:ErrorAction']='Stop'


function Check-Command($cmdname)
{
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

$ErrorActionPreference = "Stop"

$checked = Read-Host "Please ensure you are running Anaconda Powershell Prompt [y/n]"

if ($checked -ine "y")
{
    exit
}

if (Check-Command -cmdname 'conda')
{
    Write-Output "Start installation"


    conda create -y -n modfire "python<3.9" "torchdata<1" pytorch-cuda=11.7 "faiss-cpu<1.8" "torchvision<1" "pytorch>=1.13,<2" -c pytorch -c nvidia

    conda activate modfire

    if ($env:CONDA_DEFAULT_ENV -ine "modfire")
    {
        Write-Output "Can't activate conda env modfire, exit."
        exit 1
    }

    conda install -y -n modfire "tensorboard<3" "rich<11" "python-lmdb<2" "pyyaml<7" "marshmallow<4" "click<9" "msgpack-python<2" packaging -c conda-forge


    $env:ADD_ENTRY = "SET"

    pip install -e .

    python ci/post_build/win_install_post_link.py $env:CONDA_PREFIX

    Write-Output "Installation done!"

    Write-Output "If you want to train models, please install NVIDIA/Apex manually."
    Write-Output "If you want to use streamlit service, please install streamlit via pip."
}
else
{
    Write-Output "conda could not be found, please ensure you've installed conda and place it in PATH."
}
