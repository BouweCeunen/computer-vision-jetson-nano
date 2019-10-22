FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-detectnet:latest

WORKDIR /app/vision

ENV FLASK_ENV=development

COPY vision/ .

RUN apt-get update && \
    apt-get install -y nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY vision/nginx.conf /etc/nginx/nginx.conf

RUN pip3 install flask imutils uwsgi

RUN chmod +x start.sh

ENTRYPOINT ["./start.sh"]