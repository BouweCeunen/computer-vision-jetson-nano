FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-detectnet:latest

WORKDIR /app/vision

ENV SWAPFILE /var/lib/swap.img
ENV SWAPSIZE 512

COPY vision/ .

RUN pip3 install bottle imutils waitress

RUN chmod +x vision.py

ENTRYPOINT ["./vision.py"]