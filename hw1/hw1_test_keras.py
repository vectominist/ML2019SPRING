import sys
import numpy as np
import math
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
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
                if float(r[i]) < 0:
                    r[i] = '0'
                test_x[n_row // p].append(ABS(float(r[i])))
            else:
                test_x[n_row // p].append(float(0))
    n_row += 1
text.close()
test_x = np.array(test_x)

# Add bias
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)

# Read model
model = load_model('model.h5')


# Calculate predicted answer
ans = []
# print(len(test_x[0]))
y_ans = model.predict(test_x)
for i in range(len(test_x)):
    ans.append(['id_' + str(i)])
    a = y_ans[i][0]
    if a < 0:
        a = 0
    ans[i].append(a)
    

text = open(ans_file, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','value'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
