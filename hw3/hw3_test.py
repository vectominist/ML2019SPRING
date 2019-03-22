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
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# Get testing file
testing_data_x = sys.argv[1]
ans_file = sys.argv[2]

data_x = []

# Read in testing data x
print('Reading testing data...')
text = open(testing_data_x, 'r', encoding='utf8')
rows = csv.reader(text, delimiter=',')
n_row = 0
for r in rows:
    if n_row != 0:
        data_x.append(r[1].split())
    n_row += 1
text.close()


# Parsing data to (x, y)
x = np.array(data_x, dtype = np.float32)
del data_x
x = x.reshape(len(x), 48, 48, 1)
x = x / 255.0

# Read model
model = load_model('model_01.h5')

# Calculate predicted answer
print('Predicting answer...')
ans = []
y_ans = model.predict(x)
a = 0
for i in range(len(x)):
    ans.append([str(i)])
    mx = 0.0
    for j in range(7):
        if y_ans[i][j] > mx:
            mx = y_ans[i][j]
            a = j
    ans[i].append(a)
    

text = open(ans_file, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()