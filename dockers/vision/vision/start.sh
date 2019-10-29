#!/bin/bash

# create swap
# mkdir -p $(dirname $SWAPFILE)
# dd if=/dev/zero of=$SWAPFILE bs=1024k count=$SWAPSIZE
# mkswap $SWAPFILE
# chmod 0600 $SWAPFILE
# swapon $SWAPFILE
# echo "$SWAPFILE none swap sw 0 0" >> /etc/fstab
# echo "created swap file"

# start inference
python3 -u vision.py