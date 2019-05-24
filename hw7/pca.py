import sys
import argparse
import math
import os
import numpy as np
from numpy import linalg as la
from skimage.io import imread, imsave

np.random.seed(6)

def load_images(image_path, image_num):
    print('Loading images...')
    if image_path[-1] != '/':
        image_path += '/'
    img_arr = np.zeros((600 * 600 * 3, image_num))
    for i in range(image_num):
        img_arr[:, i] = np.reshape(imread(image_path + str(i) + '.jpg').astype(np.float32) / 255.0, 600 * 600 * 3).transpose()
    # 1080000 * 415 matrix
    return img_arr

def save_image(image_path, x):
    x -= np.min(x)
    x /= np.max(x)
    x = (x * 255).astype(np.uint8)
    x = np.reshape(x, (600, 600 , 3))
    imsave(image_path, x)

def PCA(x, eigens):
    print('Training PCA...')
    # x :             (1080000, 415)
    # U :             (1080000, K)
    # S :             (K, )
    # V :             (K, 415)
    U, S, V = la.svd(x, full_matrices=False)
    print('Finished calculating SVD...', flush=True)
    U = U[:, 0:eigens]
    _s = S.copy()
    S = S[:eigens]
    S = np.diag(S)
    V = V[0:eigens, :]

    return U, _s, S @ V

def reconstruct(U, C, img_avg, index, img_path):
    pic = U @ C[:, index]
    save_image(img_path, pic + img_avg)

def main(args):
    images_x = load_images(args.image_path, args.image_num)
    x_avg = np.average(images_x, axis=1)
    for i in range(args.image_num):
        images_x[:, i] -= x_avg
    print(images_x.shape)

    if not os.path.exists('models'):
        os.mkdir('models')
    
    U = None
    C = None
    _s = None
    if args.trained == True:
        U = np.load(args.model_path_U)
        C = np.load(args.model_path_C)
        # reconst_images = (U @ C) + x_avg
    else:
        U, _s, C = PCA(images_x, args.eigens)
        # reconst_images += x_avg
        np.save('models/model_U', U)
        np.save('models/model_C', C)
    
    if args.eigen_face == True:
        if not os.path.exists('pic'):
            os.mkdir('pic')
        for i in range(args.eigens):
            save_image('pic/eigen_face' + str(i) + '.jpg', U[:, i].transpose())
        save_image('pic/avg_face.jpg', x_avg)
        _s /= np.sum(_s)
        for i in range(args.eigens):
            print(_s[i])

    if args.input_image != '':
        index = int(args.input_image.split('.')[0])
        reconstruct(U, C, x_avg, index, args.reconst_image)

if __name__ == '__main__':
    # Easy run :       python pca.py Aberdeen/ --eigen_face 1
    # Training :       python pca.py Aberdeen/
    # Testing :        python pca.py Aberdeen/ --trained 1 --input_image 87.jpg --reconst_image pic/87_reconstruct.jpg
    # Eigen-Face :     python pca.py Aberdeen/ --trained 1 --eigen_face 1
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Folder of images')
    
    parser.add_argument('--eigens', default=5, type=int)
    parser.add_argument('--image_num', default=415, type=int)
    parser.add_argument('--trained', default=False, type=bool)
    parser.add_argument('--model_path_U', default='models/model_U.npy', type=str)
    parser.add_argument('--model_path_C', default='models/model_C.npy', type=str)
    parser.add_argument('--input_image', default='', type=str)
    parser.add_argument('--reconst_image', default='', type=str)
    parser.add_argument('--eigen_face', default=False, type=bool)
    args = parser.parse_args()

    main(args)
    