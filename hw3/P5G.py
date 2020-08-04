import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv

ytrain = []
Xtrain = []
with open('train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        if len(row) == 3:
            ytrain.append(int(row[0]))
            Xtrain.append([float(row[1]), float(row[2])])

ytrain = np.array(ytrain)
Xtrain = np.array(Xtrain)

ytest = []
Xtest = []
with open('test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        if len(row) == 3:
            ytest.append(int(row[0]))
            Xtest.append([float(row[1]), float(row[2])])

Xtest = np.array(Xtest)
ytest = np.array(ytest)
_p = [1, 2, 3, 4]
Cvals = np.logspace(-4, 2, 25, base=10)
gamma_vals = np.logspace(-2, 2, 25, base=10)

for p in _p:
    c_val = -1
    g_val = -1
    best_train = -1
    best_test = -1
    for c in Cvals:
        for g in gamma_vals:
            clf = SVC(C=c, kernel='rbf', degree=p, gamma=g, coef0=1.0, shrinking=True, probability=False, max_iter=1000)
            clf.fit(Xtrain, ytrain)
            if clf.score(Xtest, ytest) > c_val:
                c_val = c
                g_val = g
                best_train = clf.score(Xtrain, ytrain)
                best_test = clf.score(Xtest, ytest)

    print('p:', p)
    print('c:', c_val)
    print('gamma:', g_val)
    print('Train:', best_train)
    print('Test:', best_test)
    print('')
