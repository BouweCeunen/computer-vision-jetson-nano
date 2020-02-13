FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt:latest

WORKDIR /app

ARG OCV_VERSION="4.1.2"
ENV DEBIAN_FRONTEND noninteractive

COPY cuda_gl_interop.h.patch .

RUN apt-get update && \
    apt-get install -y python3-protobuf python3-numpy python3-matplotlib && \
    apt-get install -y --no-install-recommends \
        build-essential lbzip2 make cmake g++ wget unzip pkg-config \
        libavcodec-dev libavformat-dev libavutil-dev libavresample-dev libswscale-dev libeigen3-dev libglew-dev libgstreamer1.0-0 libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools \
        gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio \
        xkb-data libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjpeg8-dev libjpeg-turbo8-dev libxine2-dev libdc1394-22-dev libv4l-dev \
        v4l-utils qv4l2 v4l2ucp libatlas-base-dev libopenblas-dev liblapack-dev liblapacke-dev gfortran libgtk2.0-dev qt5-default libqt5opengl5-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/opencv/opencv/archive/$OCV_VERSION.zip -O opencv-$OCV_VERSION.zip && unzip opencv-$OCV_VERSION.zip && rm opencv-$OCV_VERSION.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/$OCV_VERSION.zip -O opencv_modules.$OCV_VERSION.zip && unzip opencv_modules.$OCV_VERSION.zip && rm opencv_modules.$OCV_VERSION.zip && \
    mkdir -p opencv-$OCV_VERSION/build

RUN patch -N /usr/local/cuda/include/cuda_gl_interop.h < cuda_gl_interop.h.patch && \
    rm cuda_gl_interop.h.patch && \
    cd opencv-$OCV_VERSION/build && \
    cmake \
        -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_PC_FILE_NAME=opencv.pc -D CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D CUDA_ARCH_BIN="5.3" \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-$OCV_VERSION/modules -D CUDA_ARCH_PTX="" -D WITH_CUBLAS=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D ENABLE_NEON=ON -D WITH_GSTREAMER=ON \
        -D WITH_LIBV4L=ON -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_SAMPLES=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE \
        -D WITH_GTK=OFF -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_DOCS=OFF -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j8 && make install && \
    cp unix-install/opencv.pc /usr/local/lib/pkgconfig && \
    rm -rf /app/opencv-$OCV_VERSION && \
    rm -rf /app/opencv_contrib-$OCV_VERSION && \
    ldconfig

ENTRYPOINT ["/bin/bash"]