FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-detectnet:latest

WORKDIR /app/vision

ARG SWAPFILE=/var/lib/swap.img
ARG SWAPSIZE=4096

COPY vision/ .

RUN pip3 install bottle imutils waitress

RUN chmod +x start.sh

ENTRYPOINT ["./start.sh"]