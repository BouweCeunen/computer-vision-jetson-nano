FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer:latest

ENV TRT_VERSION 5.1.6
ENV TRT_VERSION_EXT 5.1.6.1
ENV TRT_PKG_VERSION=${INF_VERSION}-1
LABEL com.nvidia.inf.version="${INF_VERSION}"

COPY graphsurgeon-tf_$TRT_VERSION-1+cuda10.0_arm64.deb .
COPY tensorrt_$TRT_VERSION_EXT-1+cuda10.0_arm64.deb .
COPY uff-converter-tf_$TRT_VERSION-1+cuda10.0_arm64.deb .

RUN dpkg -i graphsurgeon-tf_$TRT_VERSION-1+cuda10.0_arm64.deb && \
    dpkg -i tensorrt_$TRT_VERSION_EXT-1+cuda10.0_arm64.deb && \
    dpkg -i uff-converter-tf_$TRT_VERSION-1+cuda10.0_arm64.deb && \
    rm graphsurgeon-tf_$TRT_VERSION-1+cuda10.0_arm64.deb && \
    rm tensorrt_$TRT_VERSION_EXT-1+cuda10.0_arm64.deb && \
    rm uff-converter-tf_$TRT_VERSION-1+cuda10.0_arm64.deb

ENTRYPOINT ["/bin/bash"]