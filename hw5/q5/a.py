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


node_range = range(5, 100, 5)
alpha_range = np.logspace(-6, 0, 4)
learn_rate = np.logspace(-2, -0.5, 4)

best_score = -1
best_nodes = -1
best_learn = -1
best_train = []
best_test = []

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


for _node in node_range:
    _best_train_score = -9999
    for _alpha in alpha_range:
        for _learn in learn_rate:
            clf = MLPClassifier(hidden_layer_sizes=_node, activation="relu", solver="sgd", learning_rate="adaptive", alpha=_alpha, learning_rate_init=_learn, max_iter=200)
            clf.fit(Xtrain, ytrain)
            score = clf.score(Xtrain, ytrain)
            if _best_train_score == -9999:
                _best_train_score = score
            if score > _best_train_score:
                _best_test_score = score
    best_train.append(_best_train_score)

for _node in node_range:
    _best_test_score = -9999
    for _alpha in alpha_range:
        for _learn in learn_rate:
            clf = MLPClassifier(hidden_layer_sizes=_node, activation="relu", solver="sgd", learning_rate="adaptive", alpha=_alpha, learning_rate_init=_learn, max_iter=200)
            clf.fit(Xtrain, ytrain)
            score = clf.score(Xtest, ytest)
            if _best_test_score == -9999:
                _best_train_score = score
            if score > _best_test_score:
                _best_test_score = score
            if score > best_score:
                best_score = score
                best_nodes = _node
                best_learn = _learn
    best_test.append(_best_test_score)

print(best_score)
print(best_nodes)
print(best_learn)

plt.plot(node_range, best_train)
plt.title("5A Training plot")
plt.xlabel("# of hidden nodes")
plt.ylabel("accuracy")
plt.show()

plt.plot(node_range, best_test)
plt.title("5A Testing plot")
plt.xlabel("# of hidden nodes")
plt.ylabel("accuracy")
plt.show()

