FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Install some basic utilities
RUN rm /etc/apt/sources.list.d/nvidia-ml.list \
 && rm /etc/apt/sources.list.d/cuda.list \
 && apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    g++ \
    libusb-1.0-0 \
    cmake \
    libssl-dev \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda update -n base -c defaults conda \
 && conda install -y python==3.8.3 \
 && conda clean -ya

# CUDA 11.1-specific steps
RUN conda install -y -c conda-forge cudatoolkit=11.1.1 \
 && conda install -y -c pytorch \
    "pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0" \
    "torchvision=0.9.1=py38_cu111" \
 && conda clean -ya

RUN pip install matplotlib scipy tensorboardX mmcv gpustat imageio ipdb future tensorboard scikit-image lpips pandas --use-feature=2020-resolver

# docker build -f Dockerfile -t torch18 .
# docker run -it --rm --gpus=all --mount type=bind,source="$(pwd)",target=/app --mount type=bind,source=/home/uss00032/data/3d,target=/home/user/data/3d --mount type=bind,source=`pwd`/data/out,target=/app/data/out -u : torch18 /bin/bash
# nelf=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --seed 1 --use_fp16 1 --config ./configs/nelf_epi_occ_stanfordlf.py
