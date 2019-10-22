FROM bouwe/jetson-nano-l4t-cuda-cudnn:latest

WORKDIR /app

ENV INF_VERSION 5.1.6

ENV INF_PKG_VERSION=${INF_VERSION}-1

LABEL com.nvidia.inf.version="${INF_VERSION}"

COPY libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb .
COPY libnvinfer5_$INF_VERSION-1+cuda10.0_arm64.deb .
COPY libnvinfer-samples_$INF_VERSION-1+cuda10.0_all.deb  .
COPY python3-libnvinfer_$INF_VERSION-1+cuda10.0_arm64.deb .
COPY python3-libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb .

RUN dpkg -i libnvinfer5_$INF_VERSION-1+cuda10.0_arm64.deb && \
    rm libnvinfer5_$INF_VERSION-1+cuda10.0_arm64.deb

RUN dpkg -i libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb && \
    rm libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb

RUN dpkg -i libnvinfer-samples_$INF_VERSION-1+cuda10.0_all.deb && \
    rm libnvinfer-samples_$INF_VERSION-1+cuda10.0_all.deb

RUN dpkg -i python3-libnvinfer_$INF_VERSION-1+cuda10.0_arm64.deb && \
    rm python3-libnvinfer_$INF_VERSION-1+cuda10.0_arm64.deb

RUN dpkg -i python3-libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb && \
    rm python3-libnvinfer-dev_$INF_VERSION-1+cuda10.0_arm64.deb

ENTRYPOINT ["/bin/bash"]