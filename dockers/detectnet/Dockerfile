FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt-opencv:latest

WORKDIR /app

ARG MODEL_DIR="./jetson-inference/data/networks/"

ARG MOBILENET_V2="SSD-Mobilenet-v2.tar.gz"
ARG MOBILENET_V2_URL="https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/${MOBILENET_V2}"

ENV TZ=Europe/Brussels
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu fonts-dejavu-core && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/dusty-nv/jetson-inference

COPY CMakePreBuild.sh ./jetson-inference/
COPY CMakeLists.txt ./jetson-inference/CMakeLists.txt
COPY ./tools/trt-console/CMakeLists.txt ./jetson-inference/tools/trt-console/CMakeLists.txt

RUN cd jetson-inference && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make && \
    make install && \
    ldconfig

RUN wget $MOBILENET_V2_URL -O $MODEL_DIR$MOBILENET_V2 && \
    tar -zxvf $MODEL_DIR$MOBILENET_V2 -C $MODEL_DIR && \
    cp /usr/bin/python3 /usr/bin/python

ENTRYPOINT ["/bin/bash"]