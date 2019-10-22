#!/usr/bin/python3

from flask import Response, Flask, render_template, send_from_directory
import threading
import imutils
import time
import cv2
import uwsgidecorators
import vision

app = Flask(__name__)

@app.route("/")
def index():
  return render_template('index.html')

@app.route("/body.css")
def body():
  return send_from_directory('templates', 'body.css')

@app.route("/feed")
def feed():
  return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate():
  if vision.outputFrame is None:
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
  (flag, encodedImage) = cv2.imencode(".jpg", cv2.cvtColor(vision.outputFrame, cv2.COLOR_BGR2RGB))
  if not flag:
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
  yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@uwsgidecorators.postfork
@uwsgidecorators.thread
def run():
  vision.run()