FROM bouwe/jetson-nano-l4t:latest

ARG CUDA_TOOLKIT="cuda-repo-l4t-10-0-local-10.0.326"
ARG CUDA_TOOLKIT_PKG="${CUDA_TOOLKIT}_1.0-1_arm64.deb"

COPY $CUDA_TOOLKIT_PKG .

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    dpkg --force-all -i ${CUDA_TOOLKIT_PKG} && \
    rm ${CUDA_TOOLKIT_PKG} && \
    apt-key add var/cuda-repo-*-local*/*.pub && \
    apt-get update && \
    apt-get install -y --allow-downgrades cuda-toolkit-10-0 libgomp1 libfreeimage-dev libopenmpi-dev openmpi-bin && \
    dpkg --purge ${CUDA_TOOLKIT} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV PATH=$PATH:$CUDA_HOME/bin

ENTRYPOINT ["/bin/bash"]