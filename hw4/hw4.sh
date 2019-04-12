#!/bin/bash

wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.0/model_Ada1_678.pth

python saliency_map.py $1 $2
python feature_map.py $1 $2
python LIME.py $1 $2

