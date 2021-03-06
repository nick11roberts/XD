FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Setup Ubuntu
RUN apt-get update --yes
RUN apt-get install -y make cmake build-essential autoconf libtool rsync ca-certificates git grep sed dpkg curl wget bzip2 unzip llvm libssl-dev libreadline-dev libncurses5-dev libncursesw5-dev libbz2-dev libsqlite3-dev zlib1g-dev mpich htop vim 

# Get Miniconda and make it the main Python interpreter
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n pytorch_env python=3.6
RUN echo "source activate pytorch_env" > ~/.bashrc
ENV PATH /opt/conda/envs/pytorch_env/bin:$PATH
ENV CONDA_DEFAULT_ENV pytorch_env
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch-nightly
#RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
RUN conda install boto3
RUN pip install scipy
RUN pip install hydra-core
RUN pip install tensorboard
RUN pip install xxhash cachetools
RUN pip install cvxpy
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install fire
RUN pip install git+https://github.com/wbaek/theconf.git

RUN git clone --single-branch --branch pt1.8 https://github.com/HazyResearch/butterfly.git /code/butterfly
#RUN git clone --single-branch --branch master https://github.com/HazyResearch/butterfly.git /code/butterfly
#RUN cd /code/butterfly && git checkout a5d7ca6fb1e0eea0480cdfe8056a040ed1f62d04 
RUN cd /code/butterfly && export FORCE_CUDA="1" && python setup.py install
RUN pip install h5py

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install
RUN mkdir -p /code/XD
ADD . /code/XD/
