import sys
import numpy as np
from numpy import linalg as la
import numpy
import math
import csv

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

# add square term
#x = np.concatenate((x,x**2), axis = 1)

# Add bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)

# Weight and some parameters
w = np.zeros(len(x[0]))
# w = np.array([0.001] * len(x[0]))
lr = 0.0005
lr_ada = 20
repeat = 25000
beta_1 = 0.5
beta_2 = 0.5
beta_1t = 0.99
beta_2t = 0.99
lmd = 10

# Start Training
x_T = x.transpose()
s_gra = np.zeros(len(x[0]))
m = np.zeros(len(x[0]))
v = np.zeros(len(x[0]))
eps_a = [1e-3] * len(x[0])
eps = np.array(eps_a)


for i in range(repeat):
    hypo = np.dot(x, w)
    loss = hypo - y
    cost = np.sum(loss ** 2) / len(x)
    cost_a = math.sqrt(cost)

    gra = np.dot(x_T, loss) + 2 * lmd * w # regularization with lambda
    #s_gra += gra ** 2
    #ada = np.sqrt(s_gra)
    m = beta_1 * m + (1.0 - beta_1) * gra
    v = beta_2 * v + (1.0 - beta_2) * (gra ** 2)
    mt = m / (1.0 - beta_1t)
    vt = v / (1.0 - beta_2t)
    beta_1t *= beta_1
    beta_2t *= beta_2

    w = w - lr * mt / (np.sqrt(vt) + eps)

    #w = w - lr_ada * gra / ada
    if i % 200 == 0:
        print('iteration: %d | Cost: %.6lf' % (i, cost_a))
    
    if cost_a < 5.40:
        print('ended at iteration: %d | Cost: %.6lf' % (i, cost_a))
        break

'''
# Stochastic Gradient Descent
repeat = 0
batch_size = 1000
for i in range(repeat):
    for j in range(len(x) - batch_size):
        hypo = np.dot(x[j:j + batch_size], w)
        loss = hypo - y[j:j + batch_size]
        cost = np.sum(loss ** 2) / len(x)
        cost_a = math.sqrt(cost)

        x_t = x[j:j + batch_size].transpose()
        gra = np.dot(x_t, loss) + 2 * lmd * w # regularization with lambda
        #s_gra += gra ** 2
        #ada = np.sqrt(s_gra)
        m = beta_1 * m + (1.0 - beta_1) * gra
        v = beta_2 * v + (1.0 - beta_2) * (gra ** 2)
        mt = m / (1.0 - beta_1t)
        vt = v / (1.0 - beta_2t)
        beta_1t *= beta_1
        beta_2t *= beta_2

        w = w - lr * mt / (np.sqrt(vt) + eps)
        #w = w - lr_ada * gra / ada
        #if (i * len(x) + j) % 200 == 0:
        #    print('iteration: %d | Cost: %.6lf' % (i * len(x) + j, cost_a))
    hypo = np.dot(x, w)
    loss = hypo - y
    cost = np.sum(loss ** 2) / len(x)
    cost_a = math.sqrt(cost)
    print('iteration: %d | Cost: %.6lf' % (i, cost_a))
'''


# Save model
np.save('model.npy', w)

print('w rms = %.6lf, std = %.6lf' % (math.sqrt(np.sum(w ** 2)) / len(w), np.std(w)))


# Check 
#xtx = np.dot(x.transpose())
hypo = np.dot(np.dot(x, la.inv(np.dot(x.transpose(), x))), x.transpose())
hypo = np.dot(hypo, y)
loss = hypo - y
cost = np.sum(loss ** 2) / len(x)
cost_a = math.sqrt(cost)
print('Min Cost: %.6lf' % (cost_a))

'''
w = np.dot(la.inv(np.dot(x.transpose(), x)), x.transpose())
w = np.dot(w, y)
'''