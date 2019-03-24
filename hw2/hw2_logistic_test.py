import sys
import numpy as np
from numpy import linalg as la
import math
import csv

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
        data_x.append([])
        for i in r:
            data_x[n_row - 1].append(float(i))
    n_row += 1
text.close()


# Parsing data to (x, y)
x = np.array(data_x)
x = normalize(x)

#print(x)
#input()


# add square term
#x = np.concatenate((x,x**2), axis = 1)

# Add bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)

# Read model
w = np.load('model.npy')

# Calculate predicted answer
print('Predicting answer...')
ans = []
for i in range(len(x)):
    ans.append([str(i + 1)])
    a = np.dot(w, x[i])
    a = np.round((1 + np.exp(-a)) ** (-1)).astype('long')
    ans[i].append(a)
    

text = open(ans_file, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
