FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-detectnet:latest

WORKDIR /app/vision

COPY vision/ .

RUN pip3 install bottle imutils bjoern

RUN chmod +x vision.py

ENTRYPOINT ["./vision.py"]