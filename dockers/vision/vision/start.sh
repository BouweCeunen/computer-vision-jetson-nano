#!/bin/bash
if [ "$ENABLE_FLASK" == "true" ]; then
  service nginx start
  service uwsgi start
fi
python3 vision.py