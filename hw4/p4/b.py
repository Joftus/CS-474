import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib.colors import ListedColormap


ytrain = []
Xtrain = []
with open('../hw3_train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        if len(row) == 3:
            ytrain.append(int(row[0]))
            Xtrain.append([float(row[1]), float(row[2])])

ytrain = np.array(ytrain)
Xtrain = np.array(Xtrain)

ytest = []
Xtest = []
with open('../hw3_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        if len(row) == 3:
            ytest.append(int(row[0]))
            Xtest.append([float(row[1]), float(row[2])])

ytest = np.array(ytest)
Xtest = np.array(Xtest)

train_acc = []
test_acc = []
best_train = -1
best_train_depth = -1
best_test = -1
best_test_depth = -1

for depth in range(1, 11):
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=depth)
    clf.fit(Xtrain, ytrain)
    acc = clf.score(Xtrain, ytrain)
    train_acc.append(acc)
    if acc > best_train:
        best_train = acc
        best_train_depth = depth

for depth in range(1, 11):
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=depth)
    clf.fit(Xtrain, ytrain)
    acc = clf.score(Xtest, ytest)
    test_acc.append(acc)
    if acc > best_test:
        best_test = acc
        best_test_depth = depth

plt.xlim(1, 10)
plt.ylim(0, 1)
plt.plot(range(1, 11), train_acc)
plt.title('4B train accuracy vs max depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.show()

plt.xlim(1, 10)
plt.ylim(0, 1)
plt.plot(range(1, 11), test_acc)
plt.title('4B test accuracy vs max depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.show()

print(best_train)
print(best_train_depth)
print(best_test)
print(best_test_depth)
