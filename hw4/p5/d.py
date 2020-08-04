'''

Make plots of testing and training accuracies for boosting as a function of the depth (use the best number of trees and the best learning rate for that particular depth).
Report the best depth and num_trees value. Use...

    clf=GradientBoostingClassifier(learning_rate=rate,n_estimators=num_trees,max_depth=depth)



For each depth, use the best number of trees and the best learning rate for that particular depth, from the sets

    depthrange=range(1,6)
    numtreerange=[50,100,150]
    learnraterange=np.logspace(-2,0,10,base=10)


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
train_best_tree = -1
train_best_rate = -1

num_trees_range = [50, 100, 150]
depth_range = range(1, 6)
learn_rate_range = np.logspace(-2, 0, 10, base=10)

for depth in depth_range:
    print("running depth %da..." % depth)
    train_parse_best = -1
    train_parse_best_tree = -1
    train_parse_best_rate = -1
    for num_trees in num_trees_range:
        for rate in learn_rate_range:
            scr = 0
            clf = GradientBoostingClassifier(learning_rate=rate, n_estimators=num_trees, max_depth=depth)
            clf.fit(Xtrain, ytrain)
            scr = clf.score(Xtrain, ytrain)
            if scr > train_parse_best:
                train_parse_best = scr
                train_parse_best_tree = num_trees
                train_parse_best_rate = rate
    train_score.append(train_parse_best)
    if train_parse_best > train_best:
        train_best = train_parse_best
        train_best_depth = depth
        train_best_tree = train_parse_best_tree
        train_best_rate = train_parse_best_rate


plt.xlim(1, 6)
plt.ylim(0, 1.1)
plt.xticks(range(1, 6))
plt.title('5D training accuracy vs. max_depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.plot(range(1, 6), train_score)
plt.show()

test_best = -1
test_best_depth = -1
test_best_tree = -1
test_best_rate = -1

for depth in depth_range:
    print("running depth %db..." % depth)
    test_parse_best = -1
    test_parse_best_tree = -1
    test_parse_best_rate = -1
    for num_trees in num_trees_range:
        for rate in learn_rate_range:
            scr = 0
            clf = GradientBoostingClassifier(learning_rate=rate, n_estimators=num_trees, max_depth=depth)
            clf.fit(Xtrain, ytrain)
            scr = clf.score(Xtest, ytest)
            if scr > test_parse_best:
                test_parse_best = scr
                test_parse_best_tree = num_trees
                test_parse_best_rate = rate
    test_score.append(test_parse_best)
    if test_parse_best > test_best:
        test_best = test_parse_best
        test_best_depth = depth
        test_best_tree = test_parse_best_tree
        test_best_rate = test_parse_best_rate

plt.xlim(1, 6)
plt.ylim(0, 1.1)
plt.xticks(range(1, 6))
plt.title('5D testing accuracy vs. max_depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.plot(range(1, 6), test_score)
plt.show()

print(train_best)
print(train_best_depth)
print(train_best_tree)
print(train_best_rate)
print(test_best)
print(test_best_depth)
print(test_best_tree)
print(test_best_rate)
