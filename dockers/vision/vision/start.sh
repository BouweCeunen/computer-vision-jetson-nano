#!/bin/bash
if [ "$ENABLE_FLASK" == "true" ]; then
  service nginx start
  service uwsgi start
else
  python3 vision.py
fi