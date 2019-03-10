import sys
import numpy as np
from numpy import linalg as la
import math
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adamax
from keras.initializers import RandomNormal
from keras import regularizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Some functions
def ABS(x):
    if x > 0:
        return x
    else:
        return -x

# Normalization
def normalize(x):
    '''
    mean = x[...,:2].mean(0)
    std = x[...,:2].std(0)
    x[...,:2] -= mean
    x[...,:2] /= std

    mean = x[...,3:6].mean(0)
    std = x[...,:3:6].std(0)
    x[...,3:6] -= mean
    x[...,3:6] /= std
    '''
    mean = x.mean(0)
    std = x.std(0)
    x -= mean
    eps_a = np.zeros(shape = std.shape) + 1e-10
    # print(eps_a)
    x /= (std + eps_a)
    return x


# Get training file
training_data_x = sys.argv[1]
training_data_y = sys.argv[2]

data_x = []
data_y = []

# Read in training data x
text = open(training_data_x, 'r', encoding='utf8')
rows = csv.reader(text, delimiter=',')
n_row = 0
for r in rows:
    if n_row != 0:
        data_x.append([])
        for i in r:
            data_x[n_row - 1].append(float(i))
    n_row += 1
text.close()

# Read in training data y
text = open(training_data_y, 'r', encoding='utf8')
rows = csv.reader(text, delimiter=',')
n_row = 0
for r in rows:
    if n_row != 0:
        for i in r:
            data_y.append(float(i))
    n_row += 1
text.close()

# Parsing data to (x, y)
x = np.array(data_x)
x = normalize(x)
# add square term
x = np.concatenate((x,x**2), axis = 1)

y = np.array(data_y)

# Train model
model = Sequential()

model.add(Dense(15, input_dim = len(x[0]), kernel_initializer=RandomNormal(seed = 0),  \
    kernel_regularizer = regularizers.l2(0.015), \
    activation = 'relu'))

for i in range(2):
    model.add(Dropout(0.20))
    model.add(Dense(88, kernel_initializer=RandomNormal(seed = 0), kernel_regularizer = regularizers.l2(0.01), activation = 'relu'))

model.add(Dense(1, kernel_initializer=RandomNormal(seed = 0), activation = 'sigmoid'))


model.summary()
model.compile(loss = 'binary_crossentropy', \
    optimizer = Adamax(lr = 0.0035, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-6, decay = 0.003), \
    metrics=['accuracy'])
check_point = ModelCheckpoint('model_best.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
# early_stopping = EarlyStopping(monitor='val_acc', patience=80, verbose=2, mode = 'max')

model.fit(x, y, verbose = 2, batch_size = 500, epochs = 500, callbacks = [check_point], validation_split = 0.15)
model.save('model.h5')
del model