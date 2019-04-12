#!/bin/bash

wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.0/model_Ada1_678.pth
wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.0/model_Adam1_671.pth
wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.0/model_SGD1_668.pth

python hw3_test2.py $1 $2

