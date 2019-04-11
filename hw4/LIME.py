import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt

def dataNormalization(x):
    mn = np.mean(x)
    sd = np.std(x)
    return (x - mn) / sd, mn, sd

def deNormalization(x, mn, sd):
    return (x * sd + mn + 1e-5) / 1.00001

# Read training data
def readfile(path):
    print("Reading File...")
    x_train = [0] * 7
    x_label = [0] * 7
    x_train_rgb = [0] * 7
    used = np.zeros(7)

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    lrt = len(raw_train)
    for i in range(lrt):
        progress = ('#' * int(float(i)/lrt*40)).ljust(40)
        print ('[%05d/%05d] | %s |' % (i+1, lrt, progress), end='\r', flush=True)
        tag = int(raw_train[i][0])
        if used[tag] == 0:
            used[tag] = 1
            tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
            tmp2 = np.array([tmp[0], tmp[0], tmp[0]])
            tmp2 = np.moveaxis(tmp2, 0, 2)
            x_train[tag] = tmp
            x_train_rgb[tag] = tmp2

            #x_label[tag] = np.zeros(7, dtype=int)
            #x_label[tag][tag] = 1
            x_label[tag] = tag
        if np.sum(used) == 7:
            break

    x_train2 = np.array(x_train, dtype=float) / 255.0
    del x_train
    x_train2, mn, sd = dataNormalization(x_train2)

    x_train_rgb2 = np.array(x_train_rgb, dtype=float) / 255.0
    del x_train_rgb
    x_train_rgb2, mn, sd = dataNormalization(x_train_rgb2)

    x_label2 = np.array(x_label, dtype=int)
    del x_label
    
    #x_train2 = torch.FloatTensor(x_train2)
    # x_train_rgb2 = torch.FloatTensor(x_train_rgb2)
    # x_label2 = torch.LongTensor(x_label2)

    return x_train2, x_label2, x_train_rgb2, mn, sd

x_train, x_label, x_train_rgb, xmn, xsd = readfile(sys.argv[1])    # 'train.csv'
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
model.load_state_dict(torch.load('model_Ada1_678.pth?dl=1'))
model.cuda()
model.eval()

# def get_n_params(mdl):
#     pp=0
#     for p in list(mdl.parameters()):
#         nn=1
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp

# print(get_n_params(model))

# two functions that lime image explainer requires
def predict(imgT):
    # Input: image tensor
    # Returns a predict function which returns the probabilities of labels ((7,) numpy array)
    #print(imgT.shape)
    imgTmp = np.expand_dims(imgT[:,:,:,0], axis=1)
    #print(imgTmp.shape)
    res = model(torch.FloatTensor(imgTmp).cuda()).cpu().data.numpy()
    res = np.exp(res) / np.sum(np.exp(res)) # sofmax ?
    return res

def segmentation(imgT):
    # Input: image numpy array
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    #print(imgT[:,:, 0].shape)
    res = slic(imgT[:,:, 0].astype(np.float64), n_segments = 150, compactness=0.1)
    #print(res.shape)
    return res

path = sys.argv[2] # 'lime_pic'
if path[-1] == '/':
    path = path[:-1]


for i in range(7):
    progress = ('#' * int(float(i))).ljust(7)
    print ('Image [%d/%d] | %s |' % (i+1, 7, progress), end='\r', flush=True)
    # print(np.max(deNormalization(x_train_rgb[i], xmn, xsd)))
    # print(np.min(deNormalization(x_train_rgb[i], xmn, xsd)))
    # plt.imsave(path + '/tmp3_' + str(i) + '.jpg', deNormalization(x_train_rgb[i], xmn, xsd))
    # Initiate explainer instance
    explainer = lime_image.LimeImageExplainer()

    # Get the explaination of an image
    np.random.seed(16)
    explaination = explainer.explain_instance(
                            image=x_train_rgb[i],
                            classifier_fn=predict,
                            segmentation_fn=segmentation,
                            batch_size=1,
                            top_labels=10,
                            random_seed=0
                        )

    # Get processed image
    image, mask = explaination.get_image_and_mask(
                                label=x_label[i],
                                positive_only=False,
                                hide_rest=False,
                                num_features=5,
                                min_weight=0.0
                            )
    #print(image.shape, mask.shape)

    # save the image
    image = deNormalization(image, xmn, xsd)
    #print(image.shape)
    #print(np.max(image))
    #print(np.min(image))
    plt.imsave(path + '/fig3_' + str(i) + '.jpg', mark_boundaries(image, mask, outline_color=None))

print ('Image [%d/%d] | %s |\n' % (7, 7, '#' * 7))
