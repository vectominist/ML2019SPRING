#!/bin/bash

wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.1/best_3_75444.h5
wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.1/best_3_75433.h5
wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.1/best_3_75332.h5
wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.1/best_3_75181.h5
wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.1/best_3G_75181.h5
wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.1/wmodel_rnn

python hw6_keras_test.py $1 $2 $3

