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

# Get testing file
testing_data = sys.argv[1]
ans_file = sys.argv[2]

# Read testing data
test_x = []
n_row = 0
text = open(testing_data, 'r', encoding='big5')
rows = csv.reader(text, delimiter=',')
for r in rows:
    if n_row % p == 0:
        test_x.append([])
        for i in range(2, 11):
            test_x[n_row // p].append(ABS(float(r[i])))
    else :
        for i in range(2, 11):
            if r[i] != 'NR':
                '''if float(r[i]) < 0:
                    r[i] = '0' '''
                test_x[n_row // p].append(ABS(float(r[i])))
            else:
                test_x[n_row // p].append(float(0))
    n_row += 1
text.close()

test_x = np.array(test_x)

# Modify a little bit of testing data
eps = 1e-5
for d in range(len(test_x)):
    for i in range(p):
        test_mean = np.mean(test_x[d][i * 9:i * 9 + 9])
        test_std = np.std(test_x[d][i * 9: i * 9 + 9])
        if test_std < eps:
            continue
        for j in range(i * 9, i * 9 + 9):
            if ABS((test_x[d][j] - test_mean) / (test_std + eps)) > 8:
                test_x[d][j] = math.floor(test_mean * 100) / 100

'''text = open('data/mod_test.csv', 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
# s.writerow(['id','value'])
for d in range(len(test_x)):
    for i in range(p):
        s.writerow(test_x[d][i * 9: i * 9 + 9])
text.close()
'''

# add square term
#test_x = np.concatenate((test_x,test_x**2), axis = 1)

# Add bias
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis = 1)

# Read model
w = np.load('model.npy')

# Calculate predicted answer
ans = []
for i in range(len(test_x)):
    ans.append(['id_' + str(i)])
    a = np.dot(w, test_x[i])
    if a < 0:
        a = 0
    ans[i].append(a)
    

text = open(ans_file, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','value'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
