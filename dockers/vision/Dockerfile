FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-detectnet:latest

WORKDIR /app/vision

COPY vision/ .

RUN apt-get update && \
    apt-get install -y libev-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install bottle imutils bjoern

RUN chmod +x vision.py

ENTRYPOINT ["./vision.py"]