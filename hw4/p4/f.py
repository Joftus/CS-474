'''
Also make plots of testing and training accuracies as a function of the depth (using the best num_trees value for that depth),
with the depth ranging from 1 to 10. Report the best depth and corresponding number of trees.
'''

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
best_train_num_trees = -1
best_train_rate = -1

best_test = -1
best_test_depth = -1
best_test_num_trees = -1
best_test_rate = -1

num_tree_range = [1, 5, 10, 25, 50, 100, 200]
learn_rate_range = np.logspace(-3,0,15,base=10)

print("\n\n\nusing train data...")
for depth in range(1, 11):
    print("running depth: %d..." % depth)
    best_acc = -1
    best_num_trees = -1
    best_rate = -1
    for num_trees in num_tree_range:
        for rate in learn_rate_range:
            clf = GradientBoostingClassifier(learning_rate=rate, n_estimators=num_trees, max_depth=depth)
            clf.fit(Xtrain, ytrain)
            acc = clf.score(Xtrain, ytrain)
            if acc > best_acc:
                best_num_trees = num_trees
                best_rate = rate
                best_acc = acc

    clf = GradientBoostingClassifier(learning_rate=best_rate, n_estimators=best_num_trees, max_depth=depth)
    clf.fit(Xtrain, ytrain)
    acc = clf.score(Xtrain, ytrain)
    train_acc.append(acc)
    if acc > best_train:
        best_train = acc
        best_train_depth = depth
        best_train_num_trees = best_num_trees
        best_train_rate = best_rate

print("\n\n\nusing test data...")
for depth in range(1, 11):
    print("running depth: %d..." % depth)
    best_acc = -1
    best_num_trees = -1
    best_rate = -1
    for num_trees in num_tree_range:
        for rate in learn_rate_range:
            clf = GradientBoostingClassifier(learning_rate=rate, n_estimators=num_trees, max_depth=depth)
            clf.fit(Xtrain, ytrain)
            acc = clf.score(Xtest, ytest)
            if acc > best_acc:
                best_num_trees = num_trees
                best_rate = rate
                best_acc = acc

    clf = GradientBoostingClassifier(learning_rate=best_rate, n_estimators=best_num_trees, max_depth=depth)
    clf.fit(Xtrain, ytrain)
    acc = clf.score(Xtest, ytest)
    test_acc.append(acc)
    if acc > best_test:
        best_test = acc
        best_test_depth = depth
        best_test_num_trees = best_num_trees
        best_test_rate = best_rate

plt.xlim(1, 10)
plt.ylim(0, 1.1)
plt.plot(range(1, 11), train_acc)
plt.title('4F train accuracy vs max depth w/ trees & rates')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.show()

plt.xlim(1, 10)
plt.ylim(0, 1)
plt.plot(range(1, 11), test_acc)
plt.title('4F test accuracy vs max depth w/ tress & rates')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.show()

print(best_train)
print(best_train_depth)
print(best_train_num_trees)
print(best_train_rate)

print(best_test)
print(best_test_depth)
print(best_test_num_trees)
print(best_test_rate)
