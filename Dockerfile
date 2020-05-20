FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ARG http_proxy
ENV http_proxy=$http_proxy
ENV https_proxy=$http_proxy
ENV HTTP_PROXY=$http_proxy
ENV HTTPS_PROXY=$http_proxy

ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.1/compat/

RUN apt-get update && apt-get install -y --fix-missing \
    nano \
    wget \
    python3.7 \
    python3-distutils \
    python-pil \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0

RUN wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py';
RUN python3.7 get-pip.py \
      --disable-pip-version-check \
      --no-cache-dir

COPY requirements.txt /tmp/requirements.txt
RUN python3.7 -m pip install -U pip && \
   pip install -r /tmp/requirements.txt

RUN mkdir /root/.deepface
RUN mkdir /root/.deepface/weights

COPY finetuning.py /app/
COPY lr_face /app/lr_face
COPY deepface /app/deepface
COPY keras_vggface /app/keras_vggface

RUN apt-get clean && rm -rf /tmp/* /var/tmp/*
