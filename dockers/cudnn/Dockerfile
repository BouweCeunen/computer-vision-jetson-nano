FROM bouwe/jetson-nano-l4t-cuda:latest

ENV CUDNN_VERSION 7.5.0.56

ENV CUDNN_PKG_VERSION=${CUDNN_VERSION}-1

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

COPY libcudnn7_$CUDNN_VERSION-1+cuda10.0_arm64.deb .
COPY libcudnn7-dev_$CUDNN_VERSION-1+cuda10.0_arm64.deb .
COPY libcudnn7-doc_$CUDNN_VERSION-1+cuda10.0_arm64.deb .

RUN dpkg -i libcudnn7_$CUDNN_VERSION-1+cuda10.0_arm64.deb && \
    rm libcudnn7_$CUDNN_VERSION-1+cuda10.0_arm64.deb

RUN dpkg -i libcudnn7-dev_$CUDNN_VERSION-1+cuda10.0_arm64.deb && \
    rm libcudnn7-dev_$CUDNN_VERSION-1+cuda10.0_arm64.deb

RUN dpkg -i libcudnn7-doc_$CUDNN_VERSION-1+cuda10.0_arm64.deb && \
    rm libcudnn7-doc_$CUDNN_VERSION-1+cuda10.0_arm64.deb

ENTRYPOINT ["/bin/bash"]