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
                x_train.append(shearAugmentation(tmp, 0.15, -0.2))
            else:
                x_train.append(shearAugmentation(tmp, -0.1, 0.2))
            
            if i % 2 == 0:
                x_train.append(scalingAugmentation(tmp, 1.25, 1.25))
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

    x_label3 = np.load('soft_target_train_new.npy')
    val_label3 = np.load('soft_target_val_new.npy')

    x_train2 = torch.FloatTensor(x_train2)
    val_data2 = torch.FloatTensor(val_data2)
    x_label2 = torch.LongTensor(x_label2)
    val_label2 = torch.LongTensor(val_label2)
    x_label3 = torch.FloatTensor(x_label3)
    val_label3 = torch.FloatTensor(val_label3)

    return x_train2, x_label2, x_label3, val_data2, val_label2, val_label3


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

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
            nn.Dropout(0.2),
            conv_dw( 32,  64, 1),
            nn.Dropout(0.2),
            conv_dw( 64, 128, 2),
            nn.Dropout(0.2),
            conv_dw(128, 256, 2),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(256, 7)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


x_train, x_label, x_KD, val_data, val_label, val_KD = readfile(sys.argv[1])    # 'train.csv'
print('\nFinished reading!')

# Wrapped as dataloader
train_set = TensorDataset(x_train, x_label, x_KD)
val_set = TensorDataset(val_data, val_label, val_KD)
print('Finished tensor dataset!')

batch_size = 512
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
print('Finished dataloader!')

# Training
model = MobileNet().cuda()
# print(model)
orig_criterion = nn.CrossEntropyLoss()
soft_criterion = nn.BCELoss(reduction='none')
err_criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
best_acc = 0.0
num_epoch = 250

Temperature = 2.5
KD_weight = 4.0
beta = 1.0

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, (imgs, orig_label, soft_label) in enumerate(train_loader):
        optimizer.zero_grad()

        train_pred = model(imgs.cuda())

        batch_loss_orig = orig_criterion(train_pred, orig_label.cuda())
        batch_loss_soft = soft_criterion(F.softmax(train_pred, dim=1), F.softmax(soft_label.cuda() / Temperature, dim=1)) * (Temperature ** 2)
        batch_loss_soft = torch.mean(batch_loss_soft, dim=1)
        KD_err = err_criterion(soft_label.cuda(), orig_label.cuda()) # error of soft target

        batch_loss = (batch_loss_orig + KD_weight * torch.dot(torch.exp(-beta * KD_err), batch_loss_soft) / len(train_pred)) / (1 + KD_weight)
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == orig_label.numpy())
        train_loss += batch_loss.item()

        progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
    
    model.eval()
    for i, (imgs, orig_label, soft_label) in enumerate(val_loader):
        val_pred = model(imgs.cuda())

        batch_loss_orig = orig_criterion(val_pred, orig_label.cuda())
        batch_loss_soft = soft_criterion(F.softmax(val_pred, dim=1), F.softmax(soft_label.cuda() / Temperature, dim=1)) * (Temperature ** 2)
        batch_loss_soft = torch.mean(batch_loss_soft, dim=1)
        KD_err = err_criterion(soft_label.cuda(), orig_label.cuda()) # error of soft target

        batch_loss = (batch_loss_orig + KD_weight * torch.dot(torch.exp(-beta * KD_err), batch_loss_soft) / len(val_pred)) / (1 + KD_weight)

        val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == orig_label.numpy())
        val_loss += batch_loss.item()

        progress = ('#' * int(float(i)/len(val_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)

    val_acc = val_acc/val_set.__len__()
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss / train_set.__len__() * val_set.__len__(), val_acc, val_loss))

    if (val_acc > best_acc):
        with open('acc.txt','w') as f:
            f.write(str(epoch)+'\t'+str(val_acc)+'\n')
        torch.save(model.state_dict(), 'model.pth')
        best_acc = val_acc
        print ('Model Saved!')
    scheduler.step(val_acc)

# Run : python hw8_train.py 