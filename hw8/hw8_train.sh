#!/bin/bash

wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.2/soft_target_train_new.npy
wget https://github.com/vectominist/ZJ_Solution_Python/releases/download/0.0.2/soft_target_val_new.npy

python hw8_train.py $1
python hw8_compress.py
mv model_compressed.pth model_large3_684_compressed.pth


