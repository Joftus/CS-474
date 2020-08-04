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

for p in _p:
    c_val = -1
    best_train = -1
    best_test = -1
    for c in Cvals:
        clf = SVC(C=c, kernel='poly', degree=p, gamma=1.0, coef0=1.0, shrinking=True, probability=False, max_iter=1000)
        clf.fit(Xtrain, ytrain)
        if clf.score(Xtest, ytest) > c_val:
            c_val = c
            best_train = clf.score(Xtrain, ytrain)
            best_test = clf.score(Xtest, ytest)

    clf = SVC(C=c_val, kernel='poly', degree=p, gamma=1.0, coef0=1.0, shrinking=True, probability=False, max_iter=1000)
    clf.fit(Xtrain, ytrain)

    h = .03
    x1_min, x1_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    x2_min, x2_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    x1mesh, x2mesh = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    cmap_light = ListedColormap(['lightblue', 'lightcoral', 'grey'])
    cmap_bold = ListedColormap(['blue', 'red', 'black'])

    Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
    Z = Z.reshape(x1mesh.shape)

    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytrain_colors = [y - 1 for y in ytrain]
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('Problem 5F decision regions and training points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    ypred = clf.predict(Xtest)
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytest_colors = [y - 1 for y in ytest]
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('Problem 5F decision regions and testing points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    print(p, ':', c_val)
    print('Train:', best_train)
    print('Test:', best_test)
    print('')
