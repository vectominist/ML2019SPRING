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
from keras import regularizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Some functions
def ABS(x):
    if x > 0:
        return x
    else:
        return -x

# Normalization
def normalize(x):
    mean = x[...,:2].mean(0)
    std = x[...,:2].std(0)
    x[...,:2] -= mean
    x[...,:2] /= std

    mean = x[...,3:6].mean(0)
    std = x[...,:3:6].std(0)
    x[...,3:6] -= mean
    x[...,3:6] /= std
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

y = np.array(data_y)

# add square term
x = np.concatenate((x,x**2), axis = 1)

# Train model
model = Sequential()
# early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=2)

model.add(Dense(100, input_dim = len(x[0]), kernel_initializer='normal',  \
    kernel_regularizer = regularizers.l2(0.05), \
    activation = 'relu'))
model.add(Dropout(0.10))
model.add(Dense(30, kernel_regularizer = regularizers.l2(0.005), activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()
model.compile(loss = 'binary_crossentropy', \
    optimizer = Adamax(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-6, decay = 0.0001), \
    metrics=['accuracy'])
check_point = ModelCheckpoint('model_best.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')

model.fit(x, y, verbose = 1, batch_size = 50, epochs = 100, callbacks = [check_point], validation_split = 0.15)
model.save('model.h5')
del model