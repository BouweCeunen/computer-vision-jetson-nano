FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-detectnet:latest

WORKDIR /app/vision

ENV FLASK_ENV=development

COPY vision/ .

RUN pip3 install flask imutils

RUN chmod +x vision.py

ENTRYPOINT ["./vision.py"]