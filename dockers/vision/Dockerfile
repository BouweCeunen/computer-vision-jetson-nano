FROM bouwe/jetson-nano-l4t-cuda-cudnn-nvinfer-tensorrt-opencv-detectnet:latest

WORKDIR /app/vision

ENV FLASK_ENV=development

COPY vision/ .

RUN apt-get update && \
    apt-get install -y nginx uwsgi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY vision/nginx.conf /etc/nginx/sites-enabled/nginx.conf
COPY vision/uwsgi.ini /etc/uwsgi/apps-enabled/uwsgi.ini

RUN pip3 install flask imutils

RUN chmod +x start.sh

ENTRYPOINT ["/bin/bash"]