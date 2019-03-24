import sys
import numpy as np
from numpy import linalg as la
import math
import csv

PI = 3.14159265358979323846264338327950288

# Multivariate Gaussian
def f_Gaussian(x, mu, covi_det_sqrt, cov_inv, pi_n2):
    pw = -0.5 * np.dot(np.dot((x - mu), cov_inv), (x - mu))
    '''print(pw, math.exp(pw), covi_det_sqrt, pi_n2)
    input()'''
    return math.exp(pw) * covi_det_sqrt / pi_n2

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

print('Reading training data...')
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

data_x = np.array(data_x)
data_x = normalize(data_x)

# Read in training data y
text = open(training_data_y, 'r', encoding='utf8')
rows = csv.reader(text, delimiter=',')
n_row = 0
for r in rows:
    if n_row != 0:
        for i in r:
            data_y.append(int(i))
    n_row += 1
text.close()

# Parsing data to (x, y)
x_0 = []
x_1 = []

for i in range(len(data_y)):
    if data_y[i] == 0:
        x_0.append(data_x[i])
    else:
        x_1.append(data_x[i])
x_0 = np.array(x_0, dtype = np.float64)
x_1 = np.array(x_1, dtype = np.float64)

del data_x


# Generative model
print('Building model...')
P_0 = len(x_0) / (len(x_0) + len(x_1))
P_1 = len(x_1) / (len(x_0) + len(x_1))

mu_0 = x_0.mean(0)
mu_1 = x_1.mean(0)

cov_0 = np.zeros((len(x_0[0]), len(x_0[0])), dtype = np.float64)
cov_1 = np.zeros((len(x_1[0]), len(x_1[0])), dtype = np.float64)

for i in range(len(x_0)):
    dif_0 = x_0[i] - mu_0
    cov_0 += np.outer(dif_0, dif_0)

for i in range(len(x_1)):
    dif_1 = x_1[i] - mu_1
    cov_1 += np.outer(dif_1, dif_1)

cov_0 /= len(x_0)
cov_1 /= len(x_1)

# share the same covariance matrix
Cov = cov_0 * P_0 + cov_1 * P_1
''' print(Cov)
input()'''

# Get testing file
testing_data_x = sys.argv[3]
ans_file = sys.argv[4]

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

print('Predicting answer...')
# Calculate predicted answer
ans = []
a = 0
cov_inv = la.inv(Cov)
covi_det_sqrt = math.sqrt(abs(la.det(cov_inv)))
pi_n2 = (2 * PI) ** (len(x_0[0]) / 2)

eps = 1e-7
for i in range(len(x)):
    ans.append([str(i + 1)])
    Px_0 = f_Gaussian(x[i], mu_0, covi_det_sqrt, cov_inv, pi_n2)
    Px_1 = f_Gaussian(x[i], mu_1, covi_det_sqrt, cov_inv, pi_n2)
    
    if Px_0 * P_0 + Px_1 * P_1 < eps:
        cnt = 0
        while Px_0 < eps and Px_1 < eps and cnt < 220:
            Px_0 *= 10.0
            Px_1 *= 10.0
            cnt += 1

    if Px_0 * P_0 > Px_1 * P_1:
        a = 0
    else:
        a = 1

    ans[i].append(a)


text = open(ans_file, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
