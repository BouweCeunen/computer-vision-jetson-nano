#!/bin/bash
if [ "$ENABLE_FLASK" == "true" ]; then
  service nginx start
  uwsgi --ini uwsgi.ini
fi
python3 vision.py