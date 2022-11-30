# ModFire
***Mod***ern ***F***ast ***I***mage ***Re***trieval

conda create -n modfire pytorch torchvision torchdata pytorch-cuda=11.7 faiss-cpu "python<3.9" -c pytorch -c nvidia
pip install -e .

CUDA_HOME='/usr/local/cuda-11' pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/


TODO: norm_weight_decay
DDP init_method, temp file random generate
