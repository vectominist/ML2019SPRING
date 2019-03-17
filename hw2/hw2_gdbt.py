import sys
import numpy as np
from numpy import linalg as la
import math
import csv
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import cross_validation, metrics
#from sklearn.grid_search import GridSearchCV



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
    '''mean = x.mean(0)
    std = x.std(0)
    x -= mean
    eps_a = np.zeros(shape = std.shape) + 1e-10
    # print(eps_a)
    x /= (std + eps_a)'''
    return x




# Get training file
training_data_x = sys.argv[1]
training_data_y = sys.argv[2]
testing_data_x = sys.argv[3]
ans_file = sys.argv[4]

data_x = []
data_y = []

# Read in training data x
print('Reading training data...')
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


del data_x
del data_y
data_x = []
data_y = []

# Start training
print('Start training...')
gbm0 = GradientBoostingClassifier(n_estimators = 130, learning_rate = 0.1, \
    min_samples_split = 200, min_samples_leaf = 20, max_depth = 8, \
        max_features='sqrt', subsample = 0.8, \
    random_state = 10, verbose = 1)
gbm0.fit(x, y)


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

#print('Predicting answer...')
# Calculate predicted answer

y_pred = gbm0.predict(x)
ans = []
a = 0
for i in range(len(x)):
    ans.append([str(i + 1)])
    if y_pred[i] > 0.5:
        a = 1
    else:
        a = 0

    ans[i].append(a)


text = open(ans_file, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
