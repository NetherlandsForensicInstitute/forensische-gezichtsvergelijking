FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV http_proxy "http://172.17.26.15:8080"
ENV https_proxy "http://172.17.26.15:8080"
ENV HTTP_PROXY "http://172.17.26.15:8080"
ENV HTTPS_PROXY "http://172.17.26.15:8080"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --fix-missing \
    nano \
    vim \
    unzip \
    supervisor \
    git \
    wget \
    python3 \
    python3-pip \
    python-pil \
    python-lxml \
    python3-tk \
    libsm6 \
    libxext6 \
    libxrender-dev

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -U pip && \
	pip install -r /tmp/requirements.txt

#RUN python3 -m pip --no-cache-dir install -r /tmp/requirements.txt
MKDIR /root/.deepface
MKDIR /root/.deepface/weights

COPY finetuning.py /app/
COPY lr_face /app/lr_face
COPY deepface /app/deepface
COPY resources/lfw /app/resources/lfw


RUN apt-get clean && rm -rf /tmp/* /var/tmp/*

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.0/compat/


#This is not needed, but allows to start multiple processes in a container
CMD ["/usr/bin/supervisord"]
