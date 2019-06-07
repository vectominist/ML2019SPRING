import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


def compress_model(mdl_path):
    # mdl_path = 'models/model_large_669.pth'
    x = torch.load(mdl_path)

    for i in x:
        x[i] = x[i].type(torch.float16)
        # print('{} : {}, type = {}'.format(i, x[i], x[i].type(torch.float16)))
        # input()

    torch.save(x, mdl_path[:-4] + '_compressed.pth')

def decompress_model(mdl_path):
    # mdl_path = 'models/model_large_669_compressed.pth'
    x = torch.load(mdl_path)
    for i in x:
        x[i] = x[i].type(torch.float32)
    return x

# models/model_large3_684.pth
compress_model('model.pth')
