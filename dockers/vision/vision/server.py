from bottle import template, static_file, response, Bottle
import threading
import imutils
import time
import cv2

outputFrame = None
lock = threading.Lock()
app = Bottle()

@app.route("/")
def index():
  return template('index', feed=app.get_url('/feed'))

@app.route("/body.css")
def body():
  return static_file('body.css', root='./views/')

@app.route("/feed")
def feed():
  response.content_type = "multipart/x-mixed-replace; boundary=frame"
  return generate()

def generate():
  global outputFrame, lock

  while True:
    with lock:
      if outputFrame is None:
        continue
      (flag, encodedImage) = cv2.imencode(".jpg", cv2.cvtColor(outputFrame, cv2.COLOR_BGR2RGB))
      if not flag:
        continue
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

def serve():
  app.run(host='0.0.0.0', port='80', server='cherrypy')