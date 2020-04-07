FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Andrea <andrea.macarulla@gmail.com>




# Install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH="/miniconda/bin:${PATH}"

# Cudatoolkit set to 9.0 in environment.yml
COPY environment.yml /tmp/environment.yml

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda env create -f /tmp/environment.yml \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=LRface
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH="${CONDA_PREFIX}/bin:${PATH}"
ENV CONDA_AUTO_UPDATE_CONDA=false

ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}


#RUN git clone https://github.com/HolmesNL/forensische-gezichtsvergelijking/tree/add_insightface_new

COPY . /LR-face
#COPY /dataset/DukeMTMC_prepare/ /Spatial-Temporal-Re-identification/dataset/DukeMTMC_prepare/
WORKDIR /LR-face

RUN pip install -r requirements.txt


CMD bash
