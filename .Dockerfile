FROM python:3.12 as base

# Install system dependencies
FROM base as sys-deps

# dependency for networkx to use graphviz
RUN apt-get update && apt-get install -y graphviz graphviz-dev git
# GPU install
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN git config --global --add safe.directory /home/research/src
# apt-get install python3-dev, apt-get install cmake

# will allow the usage if selenium in docker container
RUN apt-get update && apt-get  install -y wget unzip \
    && wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt install -y ./google-chrome-stable_current_amd64.deb && rm google-chrome-stable_current_amd64.deb \
    && apt-get clean

# Install python dependencies
FROM sys-deps as py-deps

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

FROM py-deps as runtime

WORKDIR /home/research/src
