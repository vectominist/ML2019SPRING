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
    '''
    if n_row % p == 9:
        # only PM2.5
        test_x.append([])

        for i in range(2, 11):
            if float(r[i]) < 0:
                r[i] = '0'
            test_x[n_row // p].append(ABS(float(r[i])))
        n_row += 1
        continue
    else:
        n_row += 1
        continue
    '''

    if n_row % p == 0:
        test_x.append([])

        for i in range(2, 11):
            '''if float(r[i]) < 0:
                r[i] = '0'''
            test_x[n_row // p].append(ABS(float(r[i])))
    else :
        for i in range(2, 11):
            if r[i] != 'NR':
                '''if float(r[i]) < 0:
                    r[i] = '0'''
                test_x[n_row // p].append(ABS(float(r[i])))
            else:
                test_x[n_row // p].append(float(0))
    n_row += 1
text.close()

test_x = np.array(test_x)

eps = 1e-6
for d in range(len(test_x)):
    # PM2.5 only
    #break
    for i in range(p):
        if i == 10 or i == 9:
            # rain fall and PM2.5
            continue
        i9 = i * 9
        #test_mean = np.mean(test_x[d][i * 9:i * 9 + 9])
        test_std = np.std(test_x[d][i9: i9 + 9])
        nowj = np.array([])
        if test_std < eps:
            continue
        for j in range(i9, i9 + 9):
            if test_x[d][10 * 9 + j - i9] > 0.01:
                # do not change data when there's rain fall
                continue
            if j == i9:
                nowj = test_x[d][i9 + 1:i9 + 9]
            elif j == i9 + 8:
                nowj = test_x[d][i9:i9 + 8]
            else:
                nowj = np.concatenate((test_x[d][i9:j], test_x[d][j + 1: i9 + 9]))
            
            now_mean = np.mean(nowj)
            now_std = np.std(nowj)
            normj = ABS(test_x[d][j] - now_mean) / (now_std + eps)
            if i == 2 and normj > 2.5:
                # CO ~ NMHC ~ NO
                if test_x[d][j] / (test_x[d][j + 9] + eps) > 2 and test_x[d][j] / (test_x[d][j + 9] + eps) < 5.1:
                    continue
                
            if i == 3 and normj > 1 and \
                    (test_x[d][j - 9] / (test_x[d][j] + eps) < 2 or test_x[d][j - 9] / (test_x[d][j] + eps) > 5.1):
                test_x[d][j] = math.floor(test_x[d][j - 9] * 100 / 3.50) / 100
                continue
            else:
                continue
                
            if i == 4 and normj > 1 and \
                    (test_x[d][j] / (test_x[d][j + 9] + eps) < 5 or test_x[d][j] / (test_x[d][j + 9] + eps) > 25):
                test_x[d][j] = 16 * test_x[d][j - 9]
                continue
            else:
                continue
                
            if i == 5 and normj > 1:
                # NOx ~ NO2
                if ABS(test_x[d][j + 9] - test_x[d][j]) < 6:
                    continue
                
            if i == 6 and normj > 1:
                # NOx ~ NO2
                test_x[d][j] = 1.15 * test_x[d][j - 9]
                continue
            else:
                continue
                
            if j == i9:
                # boundary
                if ABS(test_x[d][j] - test_x[d][j + 1]) / (test_std + eps) < 1.5:
                    continue
                if normj > 2.5:
                    test_x[d][j] = math.floor((2 * test_x[d][j + 1] - test_x[d][j + 2]) * 100) / 100
            elif j == i9 + 8:
                # boundary
                if ABS(test_x[d][j] - test_x[d][j - 1]) / (test_std + eps) < 1.5:
                    continue
                
                if normj > 5.5:
                    test_x[d][j] = math.floor((2 * test_x[d][j - 1] - test_x[d][j - 2]) * 100) / 100
            else:
                # not boundary -> (left + right) / 2
                if ABS(test_x[d][j] - test_x[d][j - 1]) / (test_std + eps) < 1.5:
                    continue
                if  normj > 5.5:
                    test_x[d][j] = math.floor((test_x[d][j - 1] + test_x[d][j + 1]) * 50) / 100




'''
# Modify a little bit of testing data
eps = 1e-6
for d in range(len(test_x)):
    for i in range(p):
        if i == 10 or i == 9:
            # rain fall and PM2.5
            continue
        i9 = i * 9
        #test_mean = np.mean(test_x[d][i * 9:i * 9 + 9])
        test_std = np.std(test_x[d][i9: i9 + 9])
        nowj = np.array([])
        if test_std < eps:
            continue
        for j in range(i9, i9 + 9):
            if test_x[d][10 * 9 + j - i9] > 0.01:
                # do not change data when there's rain fall
                continue
            if j == i9:
                nowj = test_x[d][i9 + 1:i9 + 9]
            elif j == i9 + 8:
                nowj = test_x[d][i9:i9 + 8]
            else:
                nowj = np.concatenate((test_x[d][i9:j], test_x[d][j + 1: i9 + 9]))
            
            now_mean = np.mean(nowj)
            now_std = np.std(nowj)
            normj = ABS(test_x[d][j] - now_mean) / (now_std + eps)
            if i == 2 and normj > 2.5:
                # CO ~ NMHC ~ NO
                if test_x[d][j] / (test_x[d][j + 9] + eps) > 1.5 and test_x[d][j] / (test_x[d][j + 9] + eps) < 5.1:
                    continue
                
            if i == 3 and normj > 1:
                if (test_x[d][j - 9] / (test_x[d][j] + eps) < 1.1 or test_x[d][j - 9] / (test_x[d][j] + eps) > 5.1):
                    test_x[d][j] = math.floor(test_x[d][j - 9] * 100 / 3.50) / 100
                continue
                
            if i == 4 and normj > 1:
                if test_x[d][j] / (test_x[d][j - 9] + eps) > 6:
                    test_x[d][j] = math.floor((test_x[d][j] / (test_x[d][j - 9] + eps) - 2) * test_x[d][j - 9] * 100) / 100
                elif (test_x[d][j] / (test_x[d][j - 9] + eps) < 1.5 or test_x[d][j] / (test_x[d][j - 9] + eps) > 5):
                    test_x[d][j] = 3 * test_x[d][j - 9]
                
                continue
                
            if i == 5 and normj > 1:
                # NOx ~ NO2
                if (test_x[d][j] / (test_x[d][j - 9] + eps)) > 0.8 and (test_x[d][j] / (test_x[d][j - 9] + eps)) < 8:
                    continue
                
            if i == 6 and normj > 1:
                # NOx ~ NO2
                if test_x[d][j] / (test_x[d][j - 9] + eps) > 5:
                    test_x[d][j] = math.floor((test_x[d][j] / (test_x[d][j - 9] + eps) - 1.8) * test_x[d][j - 9] * 100) / 100
                elif (test_x[d][j] / (test_x[d][j - 9] + eps)) > 3 or (test_x[d][j] / (test_x[d][j - 9] + eps)) < 1:
                    test_x[d][j] = 1.20 * test_x[d][j - 9]
                continue

            if j == i9:
                # boundary
                if ABS(test_x[d][j] - test_x[d][j + 1]) / (test_std + eps) < 1.5:
                    continue
                if normj > 2.5:
                    test_x[d][j] = math.floor((2 * test_x[d][j + 1] - test_x[d][j + 2]) * 100) / 100
            elif j == i9 + 8:
                # boundary
                if ABS(test_x[d][j] - test_x[d][j - 1]) / (test_std + eps) < 1.5:
                    continue
                
                if normj > 5.5:
                    test_x[d][j] = math.floor((2 * test_x[d][j - 1] - test_x[d][j - 2]) * 100) / 100
            else:
                # not boundary -> (left + right) / 2
                if ABS(test_x[d][j] - test_x[d][j - 1]) / (test_std + eps) < 1.5:
                    continue
                if  normj > 5.5:
                    test_x[d][j] = math.floor((test_x[d][j - 1] + test_x[d][j + 1]) * 50) / 100
'''

text = open('data/mod_test.csv', 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
# s.writerow(['id','value'])
for d in range(len(test_x)):
    for i in range(p):
        s.writerow(test_x[d][i * 9: i * 9 + 9])
text.close()


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
