import sys
import numpy as np
from numpy import linalg as la
import math
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
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



# Get training file
training_data_x = sys.argv[1]


data_x = []
data_y = []

# Read in training data x
print('Reading training data...')
text = open(training_data_x, 'r', encoding='utf8')
rows = csv.reader(text, delimiter=',')
n_row = 0
for r in rows:
    if n_row != 0:
        data_x.append(r[1].split())
        data_y.append([])
        for i in range(7):
            if i == int(r[0]):
                data_y[n_row - 1].append(1)
            else:
                data_y[n_row - 1].append(0)
        
    n_row += 1
text.close()



# Parsing data to (x, y)
x = np.array(data_x, dtype = np.float32)
del data_x
x = x.reshape(len(x), 48, 48, 1)
x = x / 255.0

y = np.array(data_y, dtype = np.float32)


del data_y

# Train model
print('Start training...')
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (48, 48, 1), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dropout(0.10))
model.add(Dense(500, kernel_initializer=RandomNormal(seed = 0), \
    kernel_regularizer = regularizers.l2(0.01), activation = 'relu'))
model.add(Dropout(0.10))
model.add(Dense(200, kernel_initializer=RandomNormal(seed = 0), \
    kernel_regularizer = regularizers.l2(0.01), activation = 'relu'))

model.add(Dense(7, activation = 'softmax'))

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

check_point = ModelCheckpoint('model_best.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
# early_stopping = EarlyStopping(monitor='val_acc', patience=80, verbose=2, mode = 'max')

model.fit(x, y, verbose = 1, batch_size = 500, epochs = 5, callbacks = [check_point], validation_split = 0.15)
model.save('model.h5')
del model