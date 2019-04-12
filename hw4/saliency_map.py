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
from skimage import color,io


def dataNormalization(x):
    mn = np.mean(x)
    sd = np.std(x)
    return (x - mn) / sd

# Read training data
def readfile(path):
    print("Reading File...")
    x_train = [0] * 7
    x_label = [0] * 7
    used = np.zeros(7)
    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    lrt = len(raw_train)
    for i in range(lrt):
        progress = ('#' * int(float(i)/lrt*40)).ljust(40)
        print ('[%05d/%05d] | %s |' % (i+1, lrt, progress), end='\r', flush=True)
        if used[int(raw_train[i][0])] == 0:
            used[int(raw_train[i][0])] = 1
            tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
            x_train[int(raw_train[i][0])] = tmp
            x_label[int(raw_train[i][0])] = raw_train[i][0]
        if np.sum(used) == 7:
            break

    x_train2 = np.array(x_train, dtype=float) / 255.0
    del x_train
    x_train2 = dataNormalization(x_train2)

    x_label2 = np.array(x_label, dtype=int)
    del x_label
    
    x_train2 = torch.FloatTensor(x_train2)
    x_label2 = torch.LongTensor(x_label2)

    return x_train2, x_label2

x_train, x_label = readfile(sys.argv[1])    # 'train.csv'
print('\nFinished reading!')



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
model = Classifier()
model.load_state_dict(torch.load('model_Ada1_678.pth'))
model.cuda()

# Compute Saliency
def compute_saliency_maps(x, y, model):
    model.eval()
    x.requires_grad_()
    y_pred = model(x.cuda())
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliency = x.grad.abs().squeeze().data
    return saliency

def show_saliency_maps(x, y, model, path):
    x_org = x.squeeze().numpy()
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(x, y, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().cpu().numpy()
    
    num_pics = x_org.shape[0]
    if path[-1] == '/':
        path = path[:-1]
    for i in range(num_pics):
        # You need to save as the correct fig names
        # plt.imsave('p3/pic_'+ str(i), x_org[i], cmap=plt.cm.gray)
        fig=plt.figure(figsize=(4, 2))
        fig.add_subplot(1, 2, 1)
        plt.imshow(x_org[i], cmap=plt.cm.gray)
        fig.add_subplot(1, 2, 2)
        plt.imshow(saliency[i], cmap=plt.cm.jet)
        plt.savefig(path + '/fig1_' + str(i) + '.jpg')
        #plt.imsave(path + '/fig1_' + str(i) + '.jpg', saliency[i], cmap=plt.cm.jet)
        #io.imsave(path + 'fig1_' + str(i), saliency[i])



show_saliency_maps(x_train[0:10], x_label[0:10], model, sys.argv[2])
