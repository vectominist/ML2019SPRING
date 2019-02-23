import sys
import numpy as np
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

# Weight and some parameters
w = np.zeros(len(x[0]))
lr = 0.0005
repeat = 10000
beta_1 = 0.5
beta_2 = 0.5
beta_1t = 0.1
beta_2t = 0.1


# Start Training
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
m = np.zeros(len(x[0]))
v = np.zeros(len(x[0]))
eps_a = [1e-8] * len(x[0])
eps = np.array(eps_a)

for i in range(repeat):
    hypo = np.dot(x, w)
    loss = hypo - y
    cost = np.sum(loss ** 2) / len(x)
    cost_a = math.sqrt(cost)

    gra = np.dot(x_t, loss)
    # s_gra += gra ** 2
    m = beta_1 * m + (1.0 - beta_1) * gra
    v = beta_2 * v + (1.0 - beta_2) * (gra ** 2)
    mt = m / (1.0 - beta_1t)
    vt = v / (1.0 - beta_2t)
    w = w - lr * mt / np.sqrt(vt)
    print('iteration: %d | Cost: %.6lf' % (i, cost_a))


# Save model
np.save('model.npy', w)

