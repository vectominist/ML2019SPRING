#!/bin/bash

wget https://www.dropbox.com/s/8vgnlim5m1vh74c/model_Ada1_678.pth?dl=1
wget https://www.dropbox.com/s/rwc2q45c3d7msb9/model_Adam1_671.pth?dl=1
wget https://www.dropbox.com/s/rgbj2fzdse4hfbp/model_SGD1_668.pth?dl=1

python hw3_test2.py $1 $2

