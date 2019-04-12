import sys
import csv
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt

def toPic(x):
    a = np.min(x)
    b = np.max(x)
    # from [a,b] -> [0, 1]
    return (x - a) / (b - a)

def showPic(x, columns, rows, pic_name):
    w=x.shape[1]
    h=x.shape[2]

    fig=plt.figure(figsize=(14, 4))
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(x[i-1], axis=2))
    plt.savefig(pic_name)
    #plt.show()

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
    i = 0
    tmp = np.array(raw_test[i, 1].split(' ')).reshape(1, 48, 48)
    x_test.append(tmp)

    x_test2 = np.array(x_test, dtype=float) / 255.0
    del x_test
    x_test2 = dataNormalization(x_test2)
    x_test2 = torch.FloatTensor(x_test2)

    return x_test2

x_test = readfile_test(sys.argv[1])    # 'test.csv'


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

def layerCal(mdl, layer_num, x):
    out = mdl.cnn[0](x)
    for i in range(1, layer_num + 1):
        out = mdl.cnn[i](out)
    return out

def layerCal2(mdl, layer_num, x):
    out = mdl.cnn(x)
    out = out.view(out.size()[0], -1)
    for i in range(layer_num + 1):
        out = mdl.fc[i](out)
    return out

# Load model
model = Classifier()
model.load_state_dict(torch.load('model_Ada1_678.pth'))
model.cuda()
model.eval()

def cnnFilterVisualiztion(mdl, layer_num, epoch, lr):
    # mdl: model
    # ly:  index of target layer

    np.random.seed(16)
    x = np.random.normal(0.0, 1.0, size=48*48).reshape(1, 48, 48)
    print(np.max(x))
    print(np.min(x))

    x_ans = []
    y_ans = []

    for i in range(64):
        progress = ('#' * int(float(i)/64*40)).ljust(40)
        print ('Filter [%03d/%03d] | %s | ' % \
                (i+1, 64, progress), end='\r', flush=True)
        
        # Generate 64 images with white noise
        x = np.random.normal(0.0, 1.0, size=48*48).reshape(1, 48, 48)
        s_grad = np.zeros(shape=x.shape)

        #print(x.shape)
        x = torch.FloatTensor(x)
        y_out = layerCal(model, layer_num, x[None].cuda())
        s_grad = torch.FloatTensor(s_grad)
        #print(y_out.shape)

        for j in range(epoch):
            x.requires_grad_()
            y_out = layerCal(model, layer_num, x[None].cuda())
            
            loss = y_out[0][i].cpu().abs().sum() - 0.15 * x.cpu().abs().sum()
            #print(loss)
            #print(y_out[0][i].cpu().abs().sum(), x.cpu().abs().sum())
            loss.backward()

            s_grad += x.grad ** 2
            x_delta = lr * x.grad / s_grad.sqrt()
            x.requires_grad_(False)

            x += x_delta

        x_ans.append(x.cpu().data.numpy())
        y_ans.append(y_out[0][i].cpu().data.numpy())
    
    print ('Filter [%03d/%03d] | %s | ' % \
        (64, 64, '#' * 40), end='\n', flush=True)
    # x: (64, 1, 48, 48)
    # y: (64, 64, 24, 24)
    return np.array(x_ans), np.array(y_ans)

def outFilterVisualiztion(mdl, epoch, lr):
    # mdl: model
    # ly:  index of target layer

    np.random.seed(16)
    x = np.random.normal(0.0, 1.0, size=48*48).reshape(1, 48, 48)
    print(np.max(x))
    print(np.min(x))

    x_ans = []
    y_ans = []
    
    for i in range(7):
        progress = ('#' * int(float(i))).ljust(7)
        print ('Filter [%03d/%03d] | %s | ' % \
                (i+1, 7, progress), end='\r', flush=True)
        
        # Generate 7 images with white noise
        x = np.random.normal(0.0, 1.0, size=48*48).reshape(1, 48, 48)
        s_grad = np.zeros(shape=x.shape)

        #print(x.shape)
        x = torch.FloatTensor(x)
        y_out = model(x[None].cuda())
        s_grad = torch.FloatTensor(s_grad)
        #print(y_out.shape)
        # y_targ = [i]
        # y_targ = torch.LongTensor(y_targ).unsqueeze(dim=1)

        for j in range(epoch):
            x.requires_grad_()
            y_out = model(x[None].cuda())
            
            #loss_func = torch.nn.CrossEntropyLoss()
            # print(y_out.shape)
            # print(x.shape)
            loss = y_out[0][i].cpu() - 0.1 * x.cpu().abs().sum()
            loss.backward()

            s_grad += x.grad ** 2
            x_delta = lr * x.grad / s_grad.sqrt()
            x.requires_grad_(False)

            x += x_delta

        x_ans.append(x.cpu().data.numpy())
    
    print ('Filter [%03d/%03d] | %s | ' % \
        (7, 7, '#' * 7), end='\n', flush=True)
    # x: (7, 1, 48, 48)
    # y: (7, 64, 24, 24)
    return np.array(x_ans)


def getConvFilter(mdl, layer_num, epoch, lr, path):
    x, y = cnnFilterVisualiztion(model, layer_num, epoch, lr)
    x = np.moveaxis(x, 1, 3)
    y = np.expand_dims(y, axis=3)

    print(x.shape, y.shape)
    print(np.max(x), np.min(x))
    print(np.max(y), np.min(y))

    x = toPic(x)
    y = toPic(y)

    #showPic(x, 16, 4, path + '/fig2_1_in.jpg')
    showPic(y, 16, 4, path + '/fig2_1.jpg')

def getOutLayer(mdl, epoch, lr, path):
    x = outFilterVisualiztion(model, epoch, lr)
    x = np.moveaxis(x, 1, 3)

    print(x.shape)
    print(np.max(x), np.min(x))

    x = toPic(x)
    showPic(x, 7, 1)
    
def getCNNImage(mdl, x, path):
    y = layerCal(model, 3, x.cuda())
    y = y[0].cpu().data.numpy()
    y = np.expand_dims(y, axis=3)
    print(y.shape)
    y = toPic(y)

    showPic(y, 16, 4, path + '/fig2_2.jpg')


layer_num = 0
epoch = 50
lr = 1.0
path = sys.argv[2]
if path[-1] == '/':
    path = path[:-1]

#getOutLayer(model, epoch, lr)

getConvFilter(model, 0, 50, 1.0, path)
getCNNImage(model, x_test, path)