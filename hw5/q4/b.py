import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
import csv

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

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


alpha_range = np.logspace(-6, 0, 5)
learn_rate_range = np.logspace(-3, -1, 3)
best_train_acc = []
best_test_acc = []

__best_nodes = -1
__best_acc = -1
__best_alpha = -1
__best_learn = -1
for _hidden in range(1, 11):
    best_train_score = -1
    best_train_alpha = -1
    best_train_learn = -1

    best_test_score = -1
    best_test_alpha = -1
    best_test_learn = -1
    for _learn in learn_rate_range:
        for _alpha in alpha_range:
            clf = MLPClassifier(hidden_layer_sizes=_hidden, activation="relu", solver="sgd", learning_rate="adaptive", alpha=_alpha, learning_rate_init=_learn, max_iter=200)
            clf.fit(Xtrain, ytrain)
            train_score = clf.score(Xtrain, ytrain)
            test_score = clf.score(Xtest, ytest)
            if train_score > best_train_score:
                best_train_score = train_score
                best_train_alpha = _alpha
                best_train_learn = _learn
            if test_score > best_test_score:
                best_test_score = test_score
                best_test_alpha = _alpha
                best_test_learn = _learn
                if test_score > __best_acc:
                    __best_acc = test_score
                    __best_nodes = _hidden
                    __best_alpha = _alpha
                    __best_learn = _learn

clf = MLPClassifier(hidden_layer_sizes=__best_nodes, activation="relu", solver="sgd", learning_rate="adaptive", alpha=__best_alpha, learning_rate_init=__best_learn, max_iter=200)
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


plt.title('4B Training data scatter plot')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
