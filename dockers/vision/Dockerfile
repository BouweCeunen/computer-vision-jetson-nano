FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-detectnet:latest

WORKDIR /app/vision

ARG SWAPFILE=/var/lib/swap.img
ARG SWAPSIZE=4096

COPY vision/ .

RUN pip3 install bottle imutils waitress

RUN mkdir -p $SWAPFILE && \
    dd if=/dev/zero of=$SWAPFILE bs=1024k count=$SWAPSIZE && \
    mkswap $SWAPFILE && \
    chmod 0600 $SWAPFILE && \
    swapon $SWAPFILE && \
    echo "$SWAPFILE none swap sw 0 0" >> /etc/fstab && \
    echo "created swap file"

RUN chmod +x vision.py

ENTRYPOINT ["./vision.py"]