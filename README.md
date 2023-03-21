<!-- Inkscape, ctrl-A, ctrl-shift-R, ctrl-shift-G, ctrl-shift-c -->


<p align="center">
  <a href="https://github.com/VL-Group/modfire#gh-light-mode-only">
    <img src="https://raw.githubusercontent.com/VL-Group/modfire/main/assets/modfire-light.svg#gh-light-mode-only" alt="modfire" title="modfire" width="45%"/>
  </a>
  <a href="https://github.com/VL-Group/modfire#gh-dark-mode-only">
    <img src="https://raw.githubusercontent.com/VL-Group/modfire/main/assets/modfire-dark.svg#gh-dark-mode-only" alt="modfire" title="modfire" width="45%"/>
  </a>
  <br/>
  <span>
    <i>a.k.a.</i> <b><i>M</i></b>odern <b><i>F</i></b>ast <b><i>I</i></b>mage <b><i>RE</i></b>trieval
  </span>
</p>


<p align="center">
  <a href="https://www.python.org/#gh-light-mode-only" target="_blank">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=000000&color=ffffff#gh-light-mode-only" alt="Python"/>
  </a>
  <a href="https://pytorch.org/#gh-light-mode-only" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=000000&color=ffffff#gh-light-mode-only" alt="PyTorch"/>
  </a>
  <a href="https://github.com/VL-Group/modfire/stargazers#gh-light-mode-only">
    <img src="https://img.shields.io/github/stars/VL-Group/modfire?logo=github&style=for-the-badge&logoColor=000000&color=dddddd&labelColor=ffffff#gh-light-mode-only" alt="Github stars"/>
  </a>
  <a href="https://github.com/VL-Group/modfire/network/members#gh-light-mode-only">
    <img src="https://img.shields.io/github/forks/VL-Group/modfire?logo=github&style=for-the-badge&logoColor=000000&color=dddddd&labelColor=ffffff#gh-light-mode-only" alt="Github forks"/>
  </a>
  <a href="https://github.com/VL-Group/modfire/blob/main/LICENSE#gh-light-mode-only">
    <img src="https://img.shields.io/github/license/VL-Group/modfire?logo=github&style=for-the-badge&logoColor=000000&color=dddddd&labelColor=ffffff#gh-light-mode-only" alt="Github license"/>
  </a>

  <a href="https://www.python.org/#gh-dark-mode-only" target="_blank">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=eeeeee&color=0d1117#gh-dark-mode-only" alt="Python"/>
  </a>
  <a href="https://pytorch.org/#gh-dark-mode-only" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=eeeeee&color=0d1117#gh-dark-mode-only" alt="PyTorch"/>
  </a>
  <a href="https://github.com/VL-Group/modfire/stargazers#gh-dark-mode-only">
    <img src="https://img.shields.io/github/stars/VL-Group/modfire?logo=github&style=for-the-badge&logoColor=eeeeee&color=333333&labelColor=0d1117#gh-dark-mode-only" alt="Github stars"/>
  </a>
  <a href="https://github.com/VL-Group/modfire/network/members#gh-dark-mode-only">
    <img src="https://img.shields.io/github/forks/VL-Group/modfire?logo=github&style=for-the-badge&logoColor=eeeeee&color=333333&labelColor=0d1117#gh-dark-mode-only" alt="Github forks"/>
  </a>
  <a href="https://github.com/VL-Group/modfire/blob/main/LICENSE#gh-dark-mode-only">
    <img src="https://img.shields.io/github/license/VL-Group/modfire?logo=github&style=for-the-badge&logoColor=eeeeee&color=333333&labelColor=0d1117#gh-dark-mode-only" alt="Github license"/>
  </a>
</p>


ModFire is a training, testing, deploying toolkit for modern fast image retrieval.


<!--ts-->
* [Motivation](#motivation)
* [Quick Start](#quick-start)
   * [Requirements](#requirements)
* [Usage](#usage)
   * [Data](#data)
   * [Training](#training)
* [Contribute to this Repository](#contribute-to-this-repository)
* [To-do List](#to-do-list)
* [References and License](#references-and-license)
   * [References](#references)
   * [Citation](#citation)
   * [Copyright](#copyright)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: runner, at: Tue Mar 21 03:18:48 UTC 2023 -->

<!--te-->



# Motivation

Evaluating performance of a hashing / quantization based retrieval model is tricky. You may meet various evaluation protocols (3 kinds of CIFAR-10, 2 kinds of NUS-WIDE, the randomness in splits also matters a lot), you may handle different backbones (what should I use? AlexNet, VGG, ResNet?), you may even fight with some legacy codes (lots of troubles with Caffe, MXNet, TensorFlow). These just mess things up.

A comprehensive benchmark is needed for all methods --- with the latest deep learning tricks. Therefore, this repo issues the whole pipeline for it, with the extensible and customizable training and evaluation configs.

# Quick Start

## Requirements

* Hardware
  * A CUDA-enabled GPU (`≥ 16GiB VRAM`)
  * `≥ 32GiB RAM`
* OS
  * All features are tested on `Ubuntu 20.04`, other platforms should also work. If not, please [file bugs](#contribute-to-this-repository).
* Software
  * A `conda` environment is highly recommended.
  * Python should be `< 3.9` due to package dependencies. The following packages would be installed if missed, but it's still recommended to install them manually to choose your preferred version.
  * `PyTorch >= 11.3` with `torchdata, torchvision`.
  * `faiss-cpu >= 1.7`


To use ModFire, a direct way is to use the PyPI package:
```bash
pip install modfire
modfire -v
```
That will prints:
```plain

                      _  __ _
  _ __ ___   ___   __| |/ _(_)_ __ ___
 | '_ ` _ \ / _ \ / _` | |_| | '__/ _ \
 | | | | | | (_) | (_| |  _| | | |  __/
 |_| |_| |_|\___/ \__,_|_| |_|_|  \___|

0.1.0
```


# Usage
## Data
Common retrieval datasets are included in the repo.
```bash
modfire dataset
```

```plain
Available datasets are:
      CIFAR10     : modfire.dataset.easy.cifar.CIFAR10
      CIFAR100    : modfire.dataset.easy.cifar.CIFAR100
      COCO        : modfire.dataset.easy.coco.COCO
      ImageNet100 : modfire.dataset.easy.imagenet100.ImageNet100
      MIRFlickr25k: modfire.dataset.easy.mirflickr25k.MIRFlickr25k
      NUS_WIDE    : modfire.dataset.easy.nuswide.NUS_WIDE
```
You could download them by just a command.
```bash
modfire dataset --root [TARGET_DIR] [DATASET_NAME]
```

## Training
```bash
modfire train [CONFIG_PATH]
```



# Contribute to this Repository
Just like other git repos, before raising issues or pull requests, please take a thorough look at [issue templates](https://github.com/VL-Group/modfire/issues/new/choose).


# To-do List
* Benchmarking site

# References and License
## References

## Citation

## Copyright

**Fonts**:
* [**Don Perry**]

**Pictures**:


**Third-party repos**:

| Repos                                                                          | License |
|-------------------------------------------------------------------------------:|---------|
| [PyTorch](https://pytorch.org/)                                                | [BSD-style](https://github.com/pytorch/pytorch/blob/master/LICENSE) |
| [Torchvision](https://pytorch.org/vision/stable/index.html)                    | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| [Apex](https://nvidia.github.io/apex/)                                         | [BSD-3-Clause](https://github.com/NVIDIA/apex/blob/master/LICENSE) |
| [Tensorboard](https://www.tensorflow.org/tensorboard)                          | [Apache-2.0](https://github.com/tensorflow/tensorboard/blob/master/LICENSE) |
| [Kornia](https://kornia.github.io/)                                            | [Apache-2.0](https://github.com/kornia/kornia/blob/master/LICENSE) |
| [rich](https://rich.readthedocs.io/en/latest/)                                 | [MIT](https://github.com/Textualize/rich/blob/master/LICENSE) |
| [python-lmdb](https://lmdb.readthedocs.io/en/release/)                         | [OpenLDAP Version 2.8](https://github.com/jnwatson/py-lmdb/blob/master/LICENSE) |
| [PyYAML](https://pyyaml.org/)                                                  | [MIT](https://github.com/yaml/pyyaml/blob/master/LICENSE) |
| [marshmallow](https://marshmallow.readthedocs.io/en/stable/)                   | [MIT](https://github.com/marshmallow-code/marshmallow/blob/dev/LICENSE) |
| [click](https://click.palletsprojects.com/)                                    | [BSD-3-Clause](https://github.com/pallets/click/blob/main/LICENSE.rst) |
| [vlutils](https://github.com/VL-Group/vlutils)                                 | [Apache-2.0](https://github.com/VL-Group/vlutils/blob/main/LICENSE) |
| [MessagePack](https://msgpack.org/)                                            | [Apache-2.0](https://github.com/msgpack/msgpack-python/blob/main/COPYING) |
| [marshmallow-jsonschema](https://github.com/fuhrysteve/marshmallow-jsonschema) | [MIT](https://github.com/fuhrysteve/marshmallow-jsonschema/blob/master/LICENSE) |
| [json-schema-for-humans](https://coveooss.github.io/json-schema-for-humans/#/) | [Apache-2.0](https://github.com/coveooss/json-schema-for-humans/blob/main/LICENSE.md) |
| [CyclicLR](https://github.com/bckenstler/CLR)                                  | [MIT](https://github.com/bckenstler/CLR/blob/master/LICENSE) |
| [Streamlit](https://streamlit.io/) | [Apache-2.0](https://github.com/streamlit/streamlit/blob/develop/LICENSE) |
| [conda](https://docs.conda.io/projects/conda/en/latest/) | [BSD 3-Clause](https://docs.conda.io/en/latest/license.html) |


<br/>
<br/>
<p align="center">
<b>
This repo is licensed under
</b>
</p>
<p align="center">
<a href="https://www.apache.org/licenses/LICENSE-2.0#gh-light-mode-only" target="_blank">
  <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/ASF_Logo-light.svg#gh-light-mode-only" alt="The Apache Software Foundation" title="The Apache Software Foundation" width="200px"/>
</a>
<a href="https://www.apache.org/licenses/LICENSE-2.0#gh-dark-mode-only" target="_blank">
<img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/ASF_Logo-light.svg#gh-dark-mode-only" alt="The Apache Software Foundation" title="The Apache Software Foundation" width="200px"/>
</a>
</p>
<p align="center">
<a href="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/LICENSE">
  <b>Apache License<br/>Version 2.0</b>
</a>
</p>

<br/>
<br/>
<br/>

<p align="center">
<a href="https://github.com/yaya-cheng#gh-dark-mode-only">
<img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/thanks.svg#gh-dark-mode-only" width="250px"/>
</a>
</p>

<!--
# ModFire
***Mod***ern ***F***ast ***I***mage ***Re***trieval

conda create -n modfire pytorch torchvision torchdata pytorch-cuda=11.7 faiss-cpu "python<3.9" -c pytorch -c nvidia
pip install -e .

CUDA_HOME='/usr/local/cuda-11' pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/


TODO: norm_weight_decay
DDP init_method, temp file random generate -->
