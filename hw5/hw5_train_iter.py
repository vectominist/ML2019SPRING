import sys
import csv
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import torchvision.models as models
from scipy import misc

# The images have to be loaded in to a range of [0, 1]
# and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
def normalization(x):
    x /= 255.0
    x[:,:,0] = (x[:,:,0] - 0.485) / 0.229
    x[:,:,1] = (x[:,:,1] - 0.456) / 0.224
    x[:,:,2] = (x[:,:,2] - 0.406) / 0.225
    
    return x

def de_normalization(x):
    x[:,:,0] = x[:,:,0] * 0.229 + 0.485
    x[:,:,1] = x[:,:,1] * 0.224 + 0.456
    x[:,:,2] = x[:,:,2] * 0.225 + 0.406
    x *= 255.0

    x = np.around(x).astype(int)
    for i in range(224):
        for j in range(224):
            for k in range(3):
                if x[i][j][k] > 255:
                    x[i][j][k] = 255
                elif x[i][j][k] < 0:
                    x[i][j][k] = 0

    return x.astype(np.uint8)

def L_inf(x_orig, x):
    return np.max(np.absolute(x - x_orig))


# Read training data
def readImage(imgPath):
    print("Reading Images...")
    
    images = []
    image_orig = []
    for i in range(200):
        progress = ('#' * int(float(i+1)/200*40)).ljust(40)
        print ('[%05d/%05d] | %s |' % (i+1, 200, progress), end='\r', flush=True)
        
        if imgPath[-1] == '/':
            imgPath = imgPath[:-1]
        
        newImgPath = imgPath + '/'
        if i < 10:
            newImgPath += '00' + str(i) + '.png'
        elif i >= 10 and i < 100:
            newImgPath += '0' + str(i) + '.png'
        else:
            newImgPath += str(i) + '.png'
        
        newImg = misc.imread(newImgPath)
        
        image_orig.append(newImg)
        
        imgTmp = normalization(newImg.copy().astype(float))
        # (224, 224, 3) -> (3, 224, 224)
        imgTmp = np.moveaxis(imgTmp,2,0)
        images.append(imgTmp)
        
    images = torch.FloatTensor(images)
    # images[0].shape = (3, 224, 224)
    
    return images, image_orig

imageSet, image_orig = readImage(sys.argv[1])    # 'data/images'
print('\nFinished reading!')


# Load pre-trained model
model = models.resnet50(pretrained=True)
model.cuda()
model.eval()

def fix(x_orig, x, eps):
    dist = x - x_orig
    # x = x_orig + dist
    for b in range(20):
        for i in range(3):
            for j in range(224):
                for k in range(224):
                    if dist[b][i][j][k] > eps:
                        x[b][i][j][k] = x_orig[b][i][j][k] + eps
                    elif dist[b][i][j][k] < -eps:
                        x[b][i][j][k] = x_orig[b][i][j][k] - eps
    return x

def fix2(dist, eps):
    if dist > eps:
        return eps
    elif dist < eps:
        return -eps
    else:
        return dist

def fix3(dist, eps):
    if dist > -1e-9:
        return eps
    else:
        return -eps

def fix4(dist, eps):
    if dist > eps * 0.1:
        return eps
    elif dist < -eps * 0.1:
        return -eps
    else:
        return 0.0

# FGSM
def train_Image(myModel, images, epochs, lr, epsR, epsG, epsB):
    # Adagrad parameters
    s_gra = torch.FloatTensor(np.zeros(shape=images[0:20].shape))

    x_orig = images.clone()
    x = images.clone()

    # First get correct label
    
    y_true = np.zeros(200, dtype=int)
    for j in range(10):
        xt = x[j * 20:j * 20 + 20]
        correct = np.argmax(myModel(xt.cuda()).cpu().data.numpy(), axis=1)
        y_true[j * 20:j * 20 + 20] = correct.copy()
    #print(y_true)
    y_true = torch.LongTensor(y_true)
    
    # Adagrad parameters
    # s_gra = torch.FloatTensor(np.zeros(shape=x.shape))
    vfix = np.vectorize(fix3)

    for i in range(epochs):
        succ_sum = 0
        for j in range(10):
            progress = ('#' * int(float(j)/10*10)).ljust(10)
            print ('[%02d/%02d] | %s |                   ' % \
            (j+1, 10, progress), end='\r', flush=True)
            
            xt = x[j * 20:j * 20 + 20]
            
            #print(xt)
            xt.requires_grad_()
            y_pred = myModel(xt.cuda())
            #print(y_pred)
            loss_func = torch.nn.CrossEntropyLoss()
            #print(y_true[j * 20:j * 20 + 20])
            loss = loss_func(y_pred, y_true[j * 20:j * 20 + 20].cuda())
            loss.backward()
            #print(xt.grad)
            s_gra += xt.grad ** 2
            x_gradT = xt.grad.clone()

            #x_delta = -lr * xt.grad / s_gra.sqrt()
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            #print(x_delta)
            xt.requires_grad_(False)
            
            ty = (np.argmax(y_pred.cpu().data.numpy(), axis=1) == y_true[j * 20:j * 20 + 20].numpy()).astype(float)
            for k in range(20):
                x_gradT[k] = x_gradT[k] * ty[k]
            
            xt -= -lr * x_gradT / s_gra.sqrt()
            

            dist = (xt - x_orig[j * 20:j * 20 + 20]).cpu().data.numpy()
            #print(dist.shape)
            dist = np.concatenate((np.expand_dims(vfix(dist[:,0,:,:], epsR), axis=1), \
                                    np.expand_dims(vfix(dist[:,1,:,:], epsG), axis=1), \
                                    np.expand_dims(vfix(dist[:,2,:,:], epsB), axis=1)), axis=1)
            #print(dist.shape)
            # x[j * 20:j * 20 + 20] = x_orig[j * 20:j * 20 + 20] + \
            #     torch.FloatTensor(vfix((xt - x_orig[j * 20:j * 20 + 20]).cpu().data.numpy(), eps))
            x[j * 20:j * 20 + 20] = x_orig[j * 20:j * 20 + 20] + torch.FloatTensor(dist)
            
            y_pred = myModel(x[j * 20:j * 20 + 20].cuda())
            succ_sum += np.sum(np.argmax(y_pred.cpu().data.numpy(), axis=1) == y_true[j * 20:j * 20 + 20].numpy())
        
        acc = 1.0 - succ_sum / 200.0
        #progress = ('#' * int(float(i + 1)/epochs*10)).ljust(10)
        print ('Epoch [%03d/%03d] | %s | Success Rate: %.3f' % \
            (i+1, epochs, '#' * 10, acc), end='\n', flush=True)
    #print(x-x_orig)
    return x.cpu().data.numpy()


# Training parameters
lr = 2.5
epochs = 10
eps = 3.00

epsR = eps / 0.229 / 255.0
epsG = eps / 0.224 / 255.0
epsB = eps / 0.225 / 255.0
#print('eps = {0}'.format(eps))

# Start training
res = train_Image(model, imageSet, epochs, lr, epsR, epsG, epsB)


total_dist = 0.0
outPath = sys.argv[2] # 'out_images'
if outPath[-1] == '/':
    outPath = outPath[:-1]


for i in range(200):
    
    # print(res[i])
    progress = ('#' * int(float(i)/200*10)).ljust(10)
    print ('Save image [%03d/%03d] | %s | L-infinity: %.6f     ' % \
        (i, 200, progress, total_dist / (i + 1)), end='\r', flush=True)
    tmpI = res[i].copy()
    # (3, 224, 224) -> (224, 224, 3)
    tmpI = np.moveaxis(tmpI, 0, 2)
    tmpI = de_normalization(tmpI)
    #print(tmpI.dtype)
    #print(tmpI, image_orig[i])
    total_dist += np.max(np.absolute(tmpI - image_orig[i])).astype(float)

    newImgPath = outPath + '/'
    if i < 10:
        newImgPath += '00' + str(i) + '.png'
    elif i >= 10 and i < 100:
        newImgPath += '0' + str(i) + '.png'
    else:
        newImgPath += str(i) + '.png'
    
    misc.imsave(newImgPath, tmpI)
    
print('L-infinity = %.6f                                             ' % (total_dist / 200.0))
