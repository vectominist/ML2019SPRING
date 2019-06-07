import sys
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from hw8_compress import decompress_model

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def dataNormalization(x):
    mn = np.mean(x)
    sd = np.std(x)
    return (x - mn) / sd


# Read testing data
def readfile_test(path):
    print("Reading File...")
    x_test = []

    raw_test = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    lrt = len(raw_test)
    for i in range(len(raw_test)):
        progress = ('#' * int(float(i)/lrt*40)).ljust(40)
        print ('[%05d/%05d] | %s |' % (i+1, lrt, progress), end='\r', flush=True)
        tmp = np.array(raw_test[i, 1].split(' ')).reshape(1, 48, 48)
        x_test.append(tmp)

    x_test2 = np.array(x_test, dtype=float) / 255.0
    del x_test
    x_test2 = dataNormalization(x_test2)
    x_test2 = torch.FloatTensor(x_test2)

    return x_test2

class MobileNet_Large(nn.Module):
    def __init__(self):
        super(MobileNet_Large, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  1,  32, 2), 
            nn.Dropout(0.1),
            conv_dw( 32,  32, 1),
            nn.Dropout(0.1),
            conv_dw( 32,  64, 1),
            nn.Dropout(0.1),
            conv_dw( 64,  64, 1),
            nn.Dropout(0.1),
            conv_dw( 64, 100, 2),
            nn.Dropout(0.2),
            conv_dw(100, 120, 2),
            nn.Dropout(0.2),
            conv_dw(120, 480, 2),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(480, 7)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 480)
        x = self.fc(x)
        return x

x_test = readfile_test(sys.argv[1])    # 'test.csv'
test_set = TensorDataset(x_test)
batch_size = 64
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


# Load model
model = MobileNet_Large()
model.load_state_dict(decompress_model('model_large3_684_compressed.pth'))
model.cuda()
model.eval()

ans = []
acc_i = 0
for i, data in enumerate(test_loader):
    pred = model(data[0].cuda())
    pred_res = np.argmax(pred.cpu().data.numpy(), axis=1)
    # print(pred_res)
    # input()
    for j in range(len(pred_res)):
        ans.append([str(acc_i + j)])
        ans[acc_i + j].append(pred_res[j])
    acc_i += len(pred_res)

text = open(sys.argv[2], 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

# python hw8_test.py data/test.csv result/prediction_large.csv
