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



# Get training file
training_data_x = sys.argv[1]
training_data_y = sys.argv[2]
testing_data = sys.argv[3]
ans_file = sys.argv[4]

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
y = np.array(data_y)

# add square term
#x = np.concatenate((x,x**2), axis = 1)

# Add bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)

# Weight and some parameters
w = np.zeros(len(x[0]))
# w = np.array([0.001] * len(x[0]))
lr = 0.000030
lr_ada = 0.005
repeat = 5000
beta_1 = 0.75
beta_2 = 0.75
beta_1t = 0.99
beta_2t = 0.99
lmd = 0

# Start Training
x_T = x.transpose()
s_gra = np.zeros(len(x[0]))
m = np.zeros(len(x[0]))
v = np.zeros(len(x[0]))
eps_a = [1e-3] * len(x[0])
eps = np.array(eps_a)
check_epochs = 50

print(x.shape)
print(w.shape)
print(y.shape)

for i in range(repeat):
    z = np.dot(x, w)
    hypo = (1 + np.exp(-z)) ** (-1)
    loss = hypo - y
    #cost = np.sum(loss ** 2) / len(x)
    #cost_a = math.sqrt(cost)

    gra = np.dot(x_T, loss) + 2 * lmd * w # regularization with lambda

    # adagrad
    s_gra += gra ** 2
    ada = np.sqrt(s_gra)

    # adam
    '''m = beta_1 * m + (1.0 - beta_1) * gra
    v = beta_2 * v + (1.0 - beta_2) * (gra ** 2)
    mt = m / (1.0 - beta_1t)
    vt = v / (1.0 - beta_2t)
    beta_1t *= beta_1
    beta_2t *= beta_2

    w = w - lr * mt / (np.sqrt(vt) + eps)
    '''

    w = w - lr_ada * gra / ada


    acc_train = 1 - np.sum(np.absolute(np.round(hypo) - y)) / len(y)

    start_test = (repeat << 5) % 32123
    z = np.dot(x[start_test:start_test + check_epochs], w)
    hypo = (1 + np.exp(-z)) ** (-1)
    acc_test = 1 - np.sum(np.absolute(np.round(hypo) - y[start_test:start_test + check_epochs])) / check_epochs
    if i % 100 == 0:
        print('iteration: %d | Loss: %.6lf | Accuracy: %.6lf' % (i, acc_train, acc_test))
    '''
    if cost_a < 5.381:
        print('ended at iteration: %d | Cost: %.6lf' % (i, cost_a))
        break
    '''


# Final cost
z = np.dot(x, w)
hypo = (1 + np.exp(-z)) ** (-1)
loss = hypo - y
acc = 1 - np.sum(np.absolute(np.round(hypo) - y)) / len(y)
print('Final Accuracy: %.6lf' % (acc))

# Save model
np.save('model.npy', w)

print('w rms = %.6lf, std = %.6lf' % (math.sqrt(np.sum(w ** 2)) / len(w), np.std(w)))
