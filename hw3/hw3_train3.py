import sys
import csv
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# Data augmentation
# Shift [0, 255] to [102, 255]
def brightnessAugentation(x):
    for i in range(48):
        for j in range(48):
            x[0][i][j] = str(int(math.floor(float(x[0][i][j]) * 0.6 + 255.0 * (1.0 - 0.6))))
    return x

def contrastAugmentation(x):
    for i in range(48):
        for j in range(48):
            k = float(x[0][i][j])
            if k >= 128:
                x[0][i][j] = str(int(math.floor(k * (((255.0 - k) / 128.0) * 0.5 + 1.0))))
            else:
                x[0][i][j] = str(int(math.floor(k * (((127.0 - k) / 127.0) * 0.5 + 0.5))))
    return x
# Shrink [0, 255] to [64, 191]
def contrastAugmentation2(x):
    for i in range(48):
        for j in range(48):
            x[0][i][j] = str(int(math.floor(float(x[0][i][j]) * 0.5 + 64.0)))
    return x

# Rotate image by theta
def rotationAugmentation(x, theta):
    tmpx = np.zeros(shape = (1, 48, 48), dtype = int)
    for i in range(48):
        for j in range(48):
            cx = 24 + int(round(( math.cos(theta) * (float(i) - 24.0) + math.sin(theta) * (float(j) - 24.0)) / math.fabs(math.cos(theta) + math.sin(theta))))
            cy = 24 + int(round((-math.sin(theta) * (float(i) - 24.0) + math.cos(theta) * (float(j) - 24.0)) / math.fabs(math.cos(theta) + math.sin(theta))))
            if cx >= 48:
                cx = 47
            elif cx < 0:
                cx = 0
            if cy >= 48:
                cy = 47
            elif cy < 0:
                cy = 0
            tmpx[0][i][j] = x[0][cx][cy]
    return tmpx

# Shear by -1 ~ 1
def shearAugmentation(x, a, b):
    tmpx = np.zeros(shape = (1, 48, 48), dtype = int)
    for i in range(48):
        for j in range(48):
            cx = 24 + int(round(( float(i - 24) - a * float(j - 24)) / (1.0 - a * b)))
            cy = 24 + int(round((-b * float(i - 24) + float(j - 24)) / (1.0 - a * b)))
            if cx >= 48:
                cx = 47
            elif cx < 0:
                cx = 0
            if cy >= 48:
                cy = 47
            elif cy < 0:
                cy = 0
            tmpx[0][i][j] = x[0][cx][cy]
    return tmpx

# Scale by 0 ~ 1.5
def scalingAugmentation(x, a, b):
    tmpx = np.zeros(shape = (1, 48, 48), dtype = int)
    for i in range(48):
        for j in range(48):
            cx = 24 + int(round(a * float(i - 24)))
            cy = 24 + int(round(b * float(j - 24)))
            if cx >= 48:
                cx = 47
            elif cx < 0:
                cx = 0
            if cy >= 48:
                cy = 47
            elif cy < 0:
                cy = 0
            tmpx[0][i][j] = x[0][cx][cy]
    return tmpx

def dataNormalization(x):
    mn = np.mean(x)
    sd = np.std(x)
    return (x - mn) / sd

def contrastEnhancement(x):
    return np.ceil(5.0396842 * np.cbrt(x - 127.0) + 127.0)


# Read training data
def readfile(path):
    print("Reading File...")
    x_train = []
    x_label = []
    val_data = []
    val_label = []

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    lrt = len(raw_train)
    for i in range(lrt):
        progress = ('#' * int(float(i)/lrt*40)).ljust(40)
        print ('[%05d/%05d] | %s |' % (i+1, lrt, progress), end='\r', flush=True)
        
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
        if (i % 10 == 0):
            val_data.append(tmp)
            val_label.append(raw_train[i][0])
        else:
            x_train.append(tmp)
            # Data augmentation
            x_train.append(np.flip(tmp, axis=2))
            if i % 2 == 0:
                x_train.append(rotationAugmentation(tmp, 0.65))
            else:
                x_train.append(rotationAugmentation(tmp, -0.65))
            
            if i % 2 == 0:
                x_train.append(shearAugmentation(tmp, 0.15, -0.3))
            else:
                x_train.append(shearAugmentation(tmp, -0.1, 0.2))
            
            if i % 2 == 0:
                x_train.append(scalingAugmentation(tmp, 1.35, 1.35))
            else:
                x_train.append(scalingAugmentation(tmp, 0.75, 0.75))
            
            x_label.append(raw_train[i][0])
            x_label.append(raw_train[i][0])
            x_label.append(raw_train[i][0])
            x_label.append(raw_train[i][0])
            x_label.append(raw_train[i][0])

    x_train2 = np.array(x_train, dtype=float) / 255.0
    del x_train
    x_train2 = dataNormalization(x_train2)

    val_data2 = np.array(val_data, dtype=float) / 255.0
    del val_data
    val_data2 = dataNormalization(val_data2)

    x_label2 = np.array(x_label, dtype=int)
    del x_label
    val_label2 = np.array(val_label, dtype=int)
    del val_label

    x_train2 = torch.FloatTensor(x_train2)
    val_data2 = torch.FloatTensor(val_data2)
    x_label2 = torch.LongTensor(x_label2)
    val_label2 = torch.LongTensor(val_label2)

    return x_train2, x_label2, val_data2, val_label2


x_train, x_label, val_data, val_label = readfile(sys.argv[1])    # 'train.csv'
print('\nFinished reading!')

# Wrapped as dataloader
train_set = TensorDataset(x_train, x_label)
val_set = TensorDataset(val_data, val_label)
print('Finished tensor dataset!')

batch_size = 512
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
print('Finished dataloader!')

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


# Training
model = Classifier().cuda()
# print(model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=5e-5)
best_acc = 0.0
num_epoch = 80

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
    
    model.eval()
    for i, data in enumerate(val_loader):
        val_pred = model(data[0].cuda())
        batch_loss = loss(val_pred, data[1].cuda())

        val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        val_loss += batch_loss.item()

        progress = ('#' * int(float(i)/len(val_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)

    val_acc = val_acc/val_set.__len__()
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss, val_acc, val_loss))

    if (val_acc > best_acc):
        with open('acc.txt','w') as f:
            f.write(str(epoch)+'\t'+str(val_acc)+'\n')
        torch.save(model.state_dict(), 'model.pth')
        best_acc = val_acc
        print ('Model Saved!')

