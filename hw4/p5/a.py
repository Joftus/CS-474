import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib.colors import ListedColormap


ytrain = []
Xtrain = []
with open('../digits-train.csv', 'r') as csvfile:
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
with open('../digits-test.csv', 'r') as csvfile:
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

train_score = []
test_score = []

train_best = -1
train_best_depth = -1

for depth in range(1, 21):
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=depth)
    clf.fit(Xtrain, ytrain)
    scr = clf.score(Xtrain, ytrain)
    train_score.append(scr)
    if scr > train_best:
        train_best = scr
        train_best_depth = depth


plt.xlim(1, 21)
plt.ylim(0, 1.1)
plt.xticks(range(1, 21))
plt.title('5A training accuracy vs. max_depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.plot(range(1, 21), train_score)
plt.show()

test_best = -1
test_best_depth = -1

for depth in range(1, 21):
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=depth)
    clf.fit(Xtrain, ytrain)
    scr = clf.score(Xtest, ytest)
    test_score.append(scr)
    if scr > test_best:
        test_best = scr
        test_best_depth = depth

plt.xlim(1, 21)
plt.ylim(0, 1)
plt.xticks(range(1, 21))
plt.title('5A testing accuracy vs. max_depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.plot(range(1, 21), test_score)
plt.show()

print(train_best)
print(train_best_depth)
print(test_best)
print(test_best_depth)
