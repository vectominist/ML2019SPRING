import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


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

x_test = readfile_test(sys.argv[1])    # 'test.csv'
test_set = TensorDataset(x_test)
batch_size = 1
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# CNN Model
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # [64, 24, 24]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]
            nn.Dropout2d(p=0.25),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]
            nn.Dropout2d(p=0.25),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),       # [256, 3, 3]
            nn.Dropout2d(p=0.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)
        )

        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# Load model
model1 = Classifier()
model1.load_state_dict(torch.load('model_Ada1_678.pth'))
model1.cuda()
model1.eval()

model2 = Classifier()
model2.load_state_dict(torch.load('model_Adam1_671.pth'))
model2.cuda()
model2.eval()

model3 = Classifier()
model3.load_state_dict(torch.load('model_SGD1_668.pth'))
model3.cuda()
model3.eval()

ans = []
for i, data in enumerate(test_loader):
    pred1 = model1(data[0].cuda())
    pred2 = model2(data[0].cuda())
    pred3 = model3(data[0].cuda())
    pred_res = np.argmax((pred1.cpu().data.numpy() + pred2.cpu().data.numpy() + pred3.cpu().data.numpy()) / 3.0, axis=1)
    # pred_res = np.argmax(pred1.cpu().data.numpy(), axis=1)
    # print(pred_res)
    # input()
    ans.append([str(i)])
    ans[i].append(pred_res[0])

text = open(sys.argv[2], 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()