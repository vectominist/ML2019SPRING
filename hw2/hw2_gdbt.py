import sys
import numpy as np
from numpy import linalg as la
import math
import csv

#import matplotlib.pylab as plt
#%matplotlib inline

from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import cross_validation, metrics
#from sklearn.model_selection import GridSearchCV

def marginal_enhance(x):
    return x


# Normalization
def normalize(x):
    mean = x[...,:2].mean(0)
    std = x[...,:2].std(0)
    x[...,:2] -= mean
    x[...,:2] /= std
    x[...,:2] = marginal_enhance(x[...,:2])

    mean = x[...,3:6].mean(0)
    std = x[...,:3:6].std(0)
    x[...,3:6] -= mean
    x[...,3:6] /= std
    x[...,3:6] = marginal_enhance(x[...,3:6])
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
raw_train = sys.argv[5]

data_x = []
data_y = []
unknown_data = [False] * 35000

# Read in training data x
print('Reading training data...')

# Check unknown
text = open(raw_train, 'r', encoding='utf8')
rows = csv.reader(text, delimiter=',')
n_row = 0
for r in rows:
    if n_row != 0:
        for i in r:
            for c in i:
                if c == '?':
                    unknown_data[n_row] = True
                    break
    n_row += 1
text.close()

text = open(training_data_x, 'r', encoding='utf8')
rows = csv.reader(text, delimiter=',')
n_row = 0
n_cnt = 0
for r in rows:
    if n_row != 0 and unknown_data[n_row] == False:
        #print(n_cnt)
        data_x.append([])
        for i in r:
            data_x[n_cnt].append(float(i))
        n_cnt += 1
    n_row += 1
text.close()



# Read in training data y
text = open(training_data_y, 'r', encoding='utf8')
rows = csv.reader(text, delimiter=',')
n_row = 0
for r in rows:
    if n_row != 0 and unknown_data[n_row] == False:
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
gbm0 = GradientBoostingClassifier(n_estimators = 160, learning_rate = 0.110, \
    min_samples_split = 200, min_samples_leaf = 50, max_depth = 8, \
    max_features='sqrt', subsample = 0.8, \
    random_state = 10, verbose = 1)
gbm0.fit(x, y)

'''
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, min_samples_leaf=20, 
      max_features='sqrt', subsample=0.8, random_state=10), 
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(x,y)
print(gsearch2.best_params_, gsearch2.best_score_)'''



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


# Calculate predicted answer
print('Predicting answer...')
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
