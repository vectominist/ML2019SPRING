import sys
import argparse
import math
import os
import time
import csv
import numpy as np
from numpy import linalg as la
import pandas as pd
from skimage.io import imread, imsave

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn import cluster

np.random.seed(0)
torch.manual_seed(0)

# !!!!!!!!!!!
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Scale by 0 ~ 1.5
def scalingAugmentation(x, a, b):
    tmpx = np.zeros(shape = (3, 32, 32), dtype = int)
    
    for k in range(3):
        for i in range(32):
            for j in range(32):
                cx = 16 + int(round(a * float(i - 16)))
                cy = 16 + int(round(b * float(j - 16)))
                if cx >= 32:
                    cx = 31
                elif cx < 0:
                    cx = 0
                if cy >= 32:
                    cy = 31
                elif cy < 0:
                    cy = 0
                tmpx[k][i][j] = x[k][cx][cy]
    return tmpx

# Rotate image by theta
def rotationAugmentation(x, theta):
    tmpx = np.zeros(shape = (3, 32, 32), dtype = int)
    for k in range(3):
        for i in range(48):
            for j in range(48):
                cx = 16 + int(round(( math.cos(theta) * (float(i) - 16.0) + math.sin(theta) * (float(j) - 16.0)) / math.fabs(math.cos(theta) + math.sin(theta))))
                cy = 16 + int(round((-math.sin(theta) * (float(i) - 16.0) + math.cos(theta) * (float(j) - 16.0)) / math.fabs(math.cos(theta) + math.sin(theta))))
                if cx >= 32:
                    cx = 31
                elif cx < 0:
                    cx = 0
                if cy >= 32:
                    cy = 31
                elif cy < 0:
                    cy = 0
                tmpx[k][i][j] = x[k][cx][cy]
    return tmpx

def load_images(image_path, image_num, augmentation=1):
    if image_path[-1] != '/':
        image_path += '/'
    
    img_arr = np.zeros((image_num * augmentation, 3, 32, 32))

    for i in range(image_num):
        progress = ('#' * int(float(i)/image_num*40)).ljust(40)
        print ('Loading images [%06d/%06d] | %s |' % (i+1, image_num, \
                progress), end='\r', flush=True)
        index = str(i + 1).zfill(6)
        img = np.moveaxis(imread(image_path + index + '.jpg').astype(np.float16) / 255.0, 2, 0)
        if augmentation == 2:
            img_arr[2 * i] = img
            # if i % 2 == 0:
            #     img_arr[2 * i + 1] = scalingAugmentation(img, 1.2, 1.2)
            # else:
            #     img_arr[2 * i + 1] = scalingAugmentation(img, 0.8, 0.8)
            img_arr[2 * i + 1] = img + np.random.normal(scale=0.005, size=(3, 32, 32))
            img_arr[2 * i + 1] -= np.min(img_arr[2 * i + 1])
            img_arr[2 * i + 1] /= np.max(img_arr[2 * i + 1])
        else:
            img_arr[i] = img
    # 40000 * 3 * 32 * 32
    return img_arr

def load_images_PCA(image_path, image_num):
    print('Loading images...')
    if image_path[-1] != '/':
        image_path += '/'
    img_arr = np.zeros((32 * 32 * 3, image_num))
    for i in range(image_num):
        progress = ('#' * int(float(i)/image_num*40)).ljust(40)
        print ('Loading images [%06d/%06d] | %s |' % (i+1, image_num, \
                progress), end='\r', flush=True)
        index = str(i + 1).zfill(6)
        img_arr[:, i] = np.reshape(imread(image_path + index + '.jpg').astype(np.float16) / 255.0, 32 * 32 * 3).transpose()
    # 3072 * 40000 matrix
    return img_arr

# def show_image(x):
#     x -= np.min(x)
#     x /= np.max(x)
#     x = np.moveaxis(x, 0, 2)
#     plt.imshow(x)
#     plt.show()

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Conv2d : (X, X, 3, 1, 1) -> same size
        # Conv2d : (X, X, 4, 2, 1) -> half size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # [64, 16, 16] = 16384
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, 1, 1), # [32, 16, 16] = 8192
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, 2, 1), # [32, 8, 8] = 2048
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, 3, 1, 1), # [16, 8, 8] = 1024
            nn.BatchNorm2d(16)
            # nn.LeakyReLU(0.2),
            # nn.Conv2d(16, 8, 3, 1, 1),  # [8, 8, 8] = 512
            # nn.BatchNorm2d(8)
        )
        # self.encoder_fc = nn.Sequential(
        #     nn.Linear(1024, 1024), 
        #     nn.LeakyReLU(0.2),
        #     nn.BatchNorm1d(1024)
        # )
        # ConvTranspose2d : (X, X, 3, 1, 1) -> same size
        # ConvTranspose2d : (X, X, 4, 2, 1) -> twice size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, 1, 1),         # [16, 8, 8] = 1024
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 32, 3, 1, 1),        # [32, 8, 8] = 2048
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),        # [32, 16, 16] = 8192
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 64, 3, 1, 1),        # [64, 16, 16] = 16384
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),         # [3, 32, 32] = 3072
            nn.Sigmoid()
        )

        # self.encoder.apply(gaussian_weights_init)
        # self.decoder.apply(gaussian_weights_init)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.encoder_fc(x.view(x.size()[0], -1))
        x = self.decoder(x.view(x.size()[0], 16, 8, 8))
        return x

def train_autoencoder(model, train_loader, epochs):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
    
    # MSELoss : lr=0.0025
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_loss = 1e10

    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        epoch_start_time = time.time()

        for i, data in enumerate(train_loader):
            img = data[0]

            train_pred = model(img.cuda())
            loss = criterion(train_pred, img.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
            print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, epochs, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
        
        print('[%03d/%03d] %2.2f sec(s) Loss: %3.6f                                           ' % \
            (epoch + 1, epochs, time.time() - epoch_start_time, train_loss))

        if (train_loss < best_loss):
            with open('loss.txt','w') as f:
                f.write(str(epoch)+'\t'+str(train_loss)+'\n')
            torch.save(model.state_dict(), 'models/autoenc_model.pth')
            best_loss = train_loss
            print ('Model Saved!')

def get_latent_vector(model, test_loader):
    l_vec = []
    model.eval()
    for i, data in enumerate(test_loader):
        progress = ('#' * int(float(i)/len(test_loader)*40)).ljust(40)
        print ('Getting Encoded Vector [%06d/%06d] | %s |' % (i*1000, len(test_loader)*1000, progress), end='\r', flush=True)
        # img = data

        # show_image(new_img[0])
        # input()

        vec = model.encoder(data[0].cuda()).cpu().data.numpy()
        # img = model(data[0].cuda()).cpu().data.numpy()
        # print(img.shape)
        for j in range(1000):
            # imsave('reconstruct/reconstruct_' + str(j) + '.jpg', (np.moveaxis(img[j], 0, 2) * 255.0).astype(np.uint8))
            # if j == 31:
            #     exit()
            l_vec.append(vec[j].flatten())
    progress = ('#' * 40).ljust(40)
    print ('Getting Encoded Vector [%06d/%06d] | %s |' % (len(test_loader)*1000, len(test_loader)*1000, progress), flush=True)
    return np.array(l_vec)

def dist(x, y):
    return euclidean(x, y)

def K_means(l_vec, eps=0.5, iters=25):
    # Using the concept of LBG Algorithm
    category = np.zeros(len(l_vec))
    # Initial condition
    v_init = np.average(l_vec, axis=0)
    # Split into two categories
    v0 = v_init * (1 + eps)
    v1 = v_init * (1 - eps)
    # init_index = 1
    # v0 = l_vec[init_index]
    # dist_arr = np.zeros(len(l_vec))
    # for i in range(len(l_vec)):
    #     dist_arr[i] = dist(v0, l_vec[i])
    # arg_arr = np.argsort(dist_arr)
    # v1 = l_vec[arg_arr[25000]]
    
    # Start iteration
    for i in range(iters):
        v0_new = np.zeros(v0.shape)
        v1_new = np.zeros(v1.shape)

        for j in range(len(l_vec)):
            progress = ('#' * int(float(j)/len(l_vec)*40)).ljust(40)
            print ('K-means [%03d/%03d] | %s |' % (i + 1, iters, progress), end='\r', flush=True)
            if dist(l_vec[j], v0) < dist(l_vec[j], v1):
                category[j] = 0
                v0_new += l_vec[j]
            else:
                category[j] = 1
                v1_new += l_vec[j]
        class0_num = len(category) - np.sum(category)
        class1_num = np.sum(category)
        v0 = v0_new / class0_num
        v1 = v1_new / class1_num
        progress = ('#' * 40).ljust(40)
        print ('K-means [%03d/%03d] | %s | Class 0 : %d , Class 1 : %d' % (i + 1, iters, progress, class0_num, class1_num))

    return v0, v1

def K_means_class(l_vec, v0, v1):
    cat_ans = []
    category = np.zeros(len(l_vec))
    for i in range(len(l_vec)):
        cat_ans.append([str(i + 1)])
        if dist(l_vec[i], v0) < dist(l_vec[i], v1):
            category[i] = 0
            cat_ans[i].append('0')
        else:
            category[i] = 1
            cat_ans[i].append('1')
    text = open('result/category.csv', 'w+')
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(['id','label'])
    for i in range(len(cat_ans)):
        s.writerow(cat_ans[i])
    text.close()
    return category


def train_enc(args):
    if not os.path.exists('models'):
        os.mkdir('models')
    
    images_x = load_images(args.image_path, args.image_num, augmentation=1)
    images_x = torch.FloatTensor(images_x)
    train_loader = DataLoader(TensorDataset(images_x), batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = Autoencoder().cuda()
    train_autoencoder(model, train_loader, args.epochs)

def train_Kmeans_PCA_test(args):
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('result'):
        os.mkdir('result')
    
    images_x = load_images(args.image_path, args.image_num, augmentation=1)
    images_x = torch.FloatTensor(images_x)
    test_loader = DataLoader(TensorDataset(images_x), batch_size=1000, shuffle=False, num_workers=0)

    model = Autoencoder()
    model.load_state_dict(torch.load('models/autoenc_model_1024_021049.pth'))
    # Loss : 0.~ (model number)
    model.cuda()
    l_vec = get_latent_vector(model, test_loader)
    # C = PCA(l_vec.transpose(), args.eigens)
    # v0, v1 = K_means(C)

    # np.save('models/v0', v0)
    # np.save('models/v1', v1)
    # category = K_means_class(C, v0, v1)
    skl_pca = PCA(n_components=args.eigens, whiten=True, random_state=87)
    pca_data = skl_pca.fit_transform(l_vec)
    sing_val = skl_pca.singular_values_
    sing_val /= np.sum(sing_val)

    text = open('result/singular_val.csv', 'w+')
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(['sing_val'])
    for i in range(len(sing_val)):
        s.writerow([str(sing_val[i])])
    text.close()

    # Test
    # pca_data = TSNE(n_components=2, random_state=87).fit_transform(pca_data)

    kmeans_fit = cluster.KMeans(n_clusters=2, random_state=87).fit(pca_data)
    category = kmeans_fit.labels_

    text = open(args.test_case, 'r', encoding='utf8')
    rows = csv.reader(text, delimiter=',')
    n_rows = 0
    ans = []
    for r in rows:
        if n_rows != 0:
            pA = int(r[1]) - 1
            pB = int(r[2]) - 1
            ans.append([str(n_rows - 1)])
            if category[pA] == category[pB]:
                ans[n_rows - 1].append(1)
            else:
                ans[n_rows - 1].append(0)
        n_rows += 1
    
    text = open(args.ans_file, 'w+')
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(['id','label'])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

def test_mode(args):
    if not os.path.exists('result'):
        os.mkdir('result')
    
    images_x = load_images(args.image_path, args.image_num)
    images_x = torch.FloatTensor(images_x)
    test_loader = DataLoader(TensorDataset(images_x), batch_size=1, shuffle=False, num_workers=0)

    model = Autoencoder()
    model.load_state_dict(torch.load('models/autoenc_model_181477.pth'))
    # Loss : 0.~ (model number)
    model.cuda()
    l_vec = get_latent_vector(model, test_loader)

    v0 = np.load('models/v0.npy')
    v1 = np.load('models/v1.npy')
    category = K_means_class(l_vec, v0, v1)

    text = open(args.test_case, 'r', encoding='utf8')
    rows = csv.reader(text, delimiter=',')
    n_rows = 0
    ans = []
    for r in rows:
        if n_rows != 0:
            pA = int(r[1]) - 1
            pB = int(r[2]) - 1
            ans.append([str(n_rows - 1)])
            if category[pA] == category[pB]:
                ans[n_rows - 1].append(1)
            else:
                ans[n_rows - 1].append(0)
        n_rows += 1
    
    text = open(args.ans_file, 'w+')
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(['id','label'])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

def visualization(args):
    vis_data = np.load('visualization.npy')
    print(vis_data.shape)
    vis_data = np.moveaxis(vis_data, 3, 1)
    print(vis_data.shape)
    vis_data = vis_data.astype(np.float16) / 255.0
    # print(vis_data[0])
    # input()

    img_data = torch.FloatTensor(vis_data)
    test_loader = DataLoader(TensorDataset(img_data), batch_size=1000, shuffle=False, num_workers=0)

    model = Autoencoder()
    model.load_state_dict(torch.load('models/autoenc_model_1024_021049.pth'))
    # Loss : 0.~ (model number)
    model.cuda()
    l_vec = get_latent_vector(model, test_loader)

    skl_pca = PCA(n_components=args.eigens, whiten=True, random_state=87)
    pca_data = skl_pca.fit_transform(l_vec)
    kmeans_fit = cluster.KMeans(n_clusters=2, random_state=87).fit(pca_data)
    category = kmeans_fit.labels_

    emb_val = TSNE(n_components=2, random_state=87).fit_transform(l_vec)

    vis_x = emb_val[:, 0]
    vis_y = emb_val[:, 1]

    # My prediction
    # plt.scatter(vis_x, vis_y, c=category, cmap=plt.cm.get_cmap("jet", 2), marker='.')
    # plt.colorbar(ticks=range(2))
    # plt.clim(0, 1)
    # plt.show()
    # Correct answer
    correct_c = np.zeros(5000, dtype=int)
    correct_c[2500:] = 1
    # plt.scatter(vis_x, vis_y, c=correct_c, cmap=plt.cm.get_cmap("jet", 2), marker='.')
    # plt.colorbar(ticks=range(2))
    # plt.clim(0, 1)
    # plt.show()

    print('acc 1 %.6f' % (np.sum(category == correct_c) / 5000))
    print('acc 2 %.6f' % (np.sum(category != correct_c) / 5000))



def main(args):
    if args.train == 0:
        train_enc(args)
    elif args.train == 1:
        train_Kmeans_PCA_test(args)
    elif args.train == 2:
        test_mode(args)
    elif args.train == 3:
        visualization(args)
    


if __name__ == '__main__':
    # Easy run :    python image_clustering.py images/ data/test_case.csv result/prediction.csv
    # Train enc :   python image_clustering.py C:\Users\HC\Documents\ML_hw7_images data/test_case.csv result/prediction.csv --train 0 --epochs 250 --batch_size 1024
    # KM PCA test : python image_clustering.py C:\Users\HC\Documents\ML_hw7_images data/test_case.csv result/prediction.csv --train 1 --eigens 1000
    # Visualize :   python image_clustering.py C:\Users\HC\Documents\ML_hw7_images data/test_case.csv result/prediction.csv --train 3 --eigens 1000
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Folder of images')
    parser.add_argument('test_case', type=str, help='Path of testing data')
    parser.add_argument('ans_file', type=str, help='Path of output answer')
    
    parser.add_argument('--train', default=0, type=int)

    # Auto-Encoder
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)

    # PCA
    parser.add_argument('--eigens', default=100, type=int)

    parser.add_argument('--image_num', default=40000, type=int)
    parser.add_argument('--show_img', default=0, type=int)
    args = parser.parse_args()

    main(args)
