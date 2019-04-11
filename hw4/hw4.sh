#!/bin/bash

wget https://www.dropbox.com/s/8vgnlim5m1vh74c/model_Ada1_678.pth?dl=1

python saliency_map.py $1 $2
python feature_map.py $1 $2
python LIME.py $1 $2

