FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    cmake \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENV MUJOCO_GL=osmesa

CMD ["/bin/bash"]