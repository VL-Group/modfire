# Tested under this image, try newer images at your own risk
# Python version = 3.8.x
FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN conda install -y "tensorboard<3" "rich<11" "python-lmdb<2" "pyyaml<7" "marshmallow<4" "click<9" "msgpack-python<2" "scipy<2" packaging "faiss-cpu<1.8" -c conda-forge -c pytorch

WORKDIR /workspace

RUN git clone https://github.com/xiaosu-zhu/modfire.git && cd modfire && PYPI_BUILDING="SET" ADD_ENTRY="SET" pip install -e .

RUN sed -i "1 s|$| -O|" "$(which modfire)"*

WORKDIR /workspace/modfire

ENTRYPOINT ["modfire"]
