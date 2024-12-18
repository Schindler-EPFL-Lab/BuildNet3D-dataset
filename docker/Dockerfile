# Define base image.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:20230309.v1

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: All commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
# ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37
ARG CUDA_ARCHITECTURES=75

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget &&\
    rm -rf /var/lib/apt/lists/*


# Download and extract Blender
RUN wget https://mirrors.dotsrc.org/blender/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
RUN tar -xf blender-4.0.2-linux-x64.tar.xz
RUN dirname /blender-4.0.2-linux-x64
RUN echo "export PATH=/blender-4.0.2-linux-x64:$PATH" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN source ~/.bashrc
SHELL ["/bin/sh", "-c"]
ENV PATH="${PATH}:/blender-4.0.2-linux-x64"
RUN echo $PATH



# Create non root user and setup environment.
RUN useradd -m -d /home/user -g root -G sudo -u 1000 user
RUN usermod -aG sudo user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new user and workdir.
USER 1000
WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

# Upgrade pip and install packages.
RUN python3.10 -m pip install --upgrade pip setuptools pathtools promise pybind11 debugpy

# Created with pip freeze -l > requirements.txt
COPY requirements.txt requirements.txt

RUN python3.10 -m pip freeze

# Install requirements on Blender's Python
COPY blender_python_library_install.py blender_python_library_install.py
RUN blender -b --python blender_python_library_install.py

# Change working directory
WORKDIR /workspace
