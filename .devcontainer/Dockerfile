#--------------------------------------------------------------
# Name:     Anaconda Environment
# Author:   Destin V
# Date:     1/6/2021
#
# Description:
#   Comes packaged with Anaconda along with basic
#   development tools.  This is a base development
#   image designed for rapid prototyping and
#   experimentation.  GPU support is provided.
#
# Note:
#   In order to get GPU support the nvidia image's
#   CUDA version must match the version on the host
#   computer.
#
#   If rendering to the screen is required then
#   execute the following prior to starting the
#   container:
#   > xhost +local:root
#
#--------------------------------------------------------------

# Source Image (this should match the CUDA version on the host)
# FROM nvidia/cuda:10.2-base-ubuntu18.04
FROM nvidia/cuda:11.4.1-base-ubuntu20.04

# Labels
LABEL maintainer="Destin V"
LABEL version="1.0"
LABEL description="Basic Anaconda development image"

# Expose ports here
# EXPOSE 8080

# Install basic packages needed for development
RUN yes Y | apt update
RUN yes Y | apt-get update
RUN yes Y | apt upgrade
RUN yes Y | apt install vim
RUN apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Setup shell
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git -p ssh-agent -p 'history-substring-search' \
    -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
    -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down'

# Setup Shell
SHELL ["/bin/zsh", "-c"]

# Miniconda requires that the path point to its python binary
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Source python versions: https://repo.anaconda.com/miniconda/
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh

RUN conda init zsh

# Setup working directory
WORKDIR /projects

# Copy folder/files into container (must be within context directory)
# Process: local files -> artifact dir -> container dir
# COPY <local_path> <container_path>

# Dependencies
RUN pip install pre-commit

# Environment
ENV SHELL /bin/zsh
ENV PYTHONPATH /directory1:/directory2
