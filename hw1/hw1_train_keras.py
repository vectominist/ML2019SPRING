import sys
import numpy as np
import math
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Some functions
def ABS(x):
    if x > 0:
        return x
    else:
        return -x



# number of data used
p = 18

# Get training file
training_data = sys.argv[1]
data = []
for i in range(p):
    data.append([])

# Read in training data
text = open(training_data, 'r', encoding='big5')
rows = csv.reader(text, delimiter=',')
n_row = 0
for r in rows:
    if n_row != 0:
        for i in range(3, 27):
            if r[i] != 'NR':
                if float(r[i]) < 0:
                    r[i] = '0'
                data[(n_row - 1) % p].append(float(r[i]))
            else:
                data[(n_row - 1) % p].append(float(0))
    n_row += 1
text.close()

# Parsing data to (x, y)
x = []
y = []
for i in range(12): # 12 months
    for j in range(471): # each month, number of 10hr data = 471
        x.append([])
        for t in range(p): # 18 kinds of data
            for hr in range(9): # every 9hr
                x[471 * i + j].append(data[t][480 * i + j + hr])
        '''
        for hr in range(9):
            x[471 * i + j].append(data[9][480 * i + j + hr])
        '''
        y.append(data[9][480 * i + j + 9]) # PM2.5
x = np.array(x)
y = np.array(y)

# Add bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)


# Some parameters
repeat = 500

# Train model
model = Sequential()
'''
model.add(Dense(12, input_dim = len(x[0]), kernel_initializer='normal', activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
'''
model.add(Dropout(0.2, input_shape = (len(x[0]), )))
model.add(Dense(30, kernel_initializer='normal', activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))


model.summary()
model.compile(loss = 'mse', optimizer = 'adam')

'''for i in range(repeat):
    cost = model.train_on_batch(x, y)
    if i % 50 == 0:
        print('iteration: %d | Cost: %.6lf' % (i, cost))
'''
model.fit(x, y, batch_size = 50, epochs = 100)
model.save('model.h5')
del model

