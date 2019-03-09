import sys
import numpy as np
from numpy import linalg as la
import math
import csv
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adamax
from keras import regularizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Normalization
def normalize(x):
    '''mean = x[...,:2].mean(0)
    std = x[...,:2].std(0)
    x[...,:2] -= mean
    x[...,:2] /= std

    mean = x[...,3:6].mean(0)
    std = x[...,:3:6].std(0)
    x[...,3:6] -= mean
    x[...,3:6] /= std'''
    mean = x.mean(0)
    std = x.std(0)
    x -= mean
    eps_a = np.zeros(shape = std.shape) + 1e-10
    # print(eps_a)
    x /= (std + eps_a)
    return x



# Get testing file
testing_data_x = sys.argv[1]
ans_file = sys.argv[2]

data_x = []

# Read in training data x
text = open(testing_data_x, 'r', encoding='utf8')
rows = csv.reader(text, delimiter=',')
n_row = 0
for r in rows:
    if n_row != 0:
        data_x.append([])
        for i in r:
            data_x[n_row - 1].append(float(i))
    n_row += 1
text.close()


# Parsing data to (x, y)
x = np.array(data_x)
x = normalize(x)
# add square term
x = np.concatenate((x,x**2), axis = 1)

# Read model
model = load_model('model.h5')

# Calculate predicted answer
ans = []
y_ans = model.predict(x)
for i in range(len(x)):
    ans.append([str(i + 1)])
    a = np.round(y_ans[i][0]).astype('long')
    ans[i].append(a)
    

text = open(ans_file, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
