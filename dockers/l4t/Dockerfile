FROM balenalib/jetson-nano-ubuntu:bionic

ARG DRIVER_PACK="Jetson-210_Linux_R32.2.1_aarch64.tbz2"
ARG POWER_MODE=0000

COPY $DRIVER_PACK .

RUN apt-get update && \
    apt-get install -y --no-install-recommends bzip2 ca-certificates curl lbzip2 sudo htop curl && \
    apt-get install -y zip git python3 python3-pip python3-numpy cmake systemd && \
    tar -xpj --overwrite -f ./${DRIVER_PACK} && \
    sed -i '/.*tar -I lbzip2 -xpmf ${LDK_NV_TEGRA_DIR}\/config\.tbz2.*/c\tar -I lbzip2 -xpm --overwrite -f ${LDK_NV_TEGRA_DIR}\/config.tbz2' ./Linux_for_Tegra/apply_binaries.sh && \
    ./Linux_for_Tegra/apply_binaries.sh -r / && \
    rm -rf ./Linux_for_Tegra && \
    rm ./${DRIVER_PACK} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install jetson-stats
    
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH}

RUN ln -s /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.32.1.0 /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so && \
    ln -s /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.32.1.0 /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.1 && \
    ln -sf /usr/lib/aarch64-linux-gnu/tegra/libGL.so /usr/lib/aarch64-linux-gnu/libGL.so && \
    ln -s /usr/lib/aarch64-linux-gnu/libcuda.so /usr/lib/aarch64-linux-gnu/libcuda.so.1 && \
    ln -sf /usr/lib/aarch64-linux-gnu/tegra-egl/libEGL.so /usr/lib/aarch64-linux-gnu/libEGL.so

RUN ln -s /etc/nvpmodel/nvpmodel_t210_jetson-nano.conf /etc/nvpmodel.conf && \
    ln -s /etc/systemd/system/nvpmodel.service /etc/systemd/system/multi-user.target.wants/nvpmodel.service && \
    mkdir /var/lib/nvpmodel && \
    echo "/etc/nvpmodel.conf" > /var/lib/nvpmodel/conf_file_path && \
    echo "pmode:${POWER_MODE} fmode:fanNull" > /var/lib/nvpmodel/status

ENTRYPOINT ["/bin/bash"]