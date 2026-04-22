FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /ws

RUN apt update && apt install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    vim \
    nano \
    python3-numpy \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

CMD ["bash"]