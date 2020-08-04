from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

ytrain = []
Xtrain = []
with open('digits-train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        x = []
        if len(row) == 48:
            ytrain.append(float(row[0]))
            for i in range(1, 48):
                x.append(float(row[i]))
            Xtrain.append(x)

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)

ytest = []
Xtest = []
with open('digits-test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        x = []
        if len(row) == 48:
            ytest.append(float(row[0]))
            for i in range(1, 48):
                x.append(float(row[i]))
            Xtest.append(x)

Xtest = np.array(Xtest)
ytest = np.array(ytest)

iter_range = range(1, 51)
node_range = range(5, 100, 5)
alpha_range = np.logspace(-6, 0, 4)

best_score = -1
best_train = []
best_test = []

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


for _iter in iter_range:
    clf = MLPClassifier(hidden_layer_sizes=40, activation="relu", solver="sgd", learning_rate="adaptive", alpha=0.0, learning_rate_init=0.1, max_iter=_iter)
    clf.partial_fit(Xtrain, ytrain, np.unique(ytrain))
    score = clf.score(Xtrain, ytrain)
    best_train.append(score)


for _iter in iter_range:
    clf = MLPClassifier(hidden_layer_sizes=40, activation="relu", solver="sgd", learning_rate="adaptive", alpha=0.0, learning_rate_init=0.1, max_iter=_iter)
    clf.partial_fit(Xtrain, ytrain, np.unique(ytrain))
    score = clf.score(Xtest, ytest)
    best_test.append(score)

plt.plot(iter_range, best_train)
plt.title("5B Training plot")
plt.xlabel("# of iter")
plt.ylabel("accuracy")
plt.show()

plt.plot(iter_range, best_test)
plt.title("5B Testing plot")
plt.xlabel("# of iter")
plt.ylabel("accuracy")
plt.show()
