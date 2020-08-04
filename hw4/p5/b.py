'''
Make plots of testing and training accuracies of bagging as a function of the depth (using the best num_trees value for that depth), with the depth ranging from 1 to 20.
Report the best depth and corresponding number of trees. Like in the previous problem, use
clf=RandomForestClassifier(bootstrap=True,n_estimators=num_trees,max_ features=None,criterion=’gini’,max_depth=depth)
3
For each depth, use the best number of trees for that particular depth, from the set numtreerange=[1,5,10,25,50,100,200].
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
num_trees_range = [1, 5, 10, 25, 50, 100, 200]

for depth in range(1, 21):
    print("running depth %da..." % depth)
    train_parse_best = -1
    train_parse_best_tree = -1
    for num_trees in num_trees_range:
        clf = RandomForestClassifier(bootstrap=True, n_estimators=num_trees, max_features = None, criterion ='gini', max_depth=depth)
        clf.fit(Xtrain, ytrain)
        scr = clf.score(Xtrain, ytrain)
        if scr > train_parse_best:
            train_parse_best = scr
            train_parse_best_tree = num_trees
    train_score.append(train_parse_best)
    if train_parse_best > train_best:
        train_best = train_parse_best
        train_best_depth = depth
        train_best_tree = train_parse_best_tree


plt.xlim(1, 21)
plt.ylim(0, 1.1)
plt.xticks(range(1, 21))
plt.title('5B training accuracy vs. max_depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.plot(range(1, 21), train_score)
plt.show()

test_best = -1
test_best_depth = -1
test_best_tree = -1

for depth in range(1, 21):
    print("running depth %db..." % depth)
    test_parse_best = -1
    test_parse_best_tree = -1
    for num_trees in num_trees_range:
        clf = RandomForestClassifier(bootstrap=True, n_estimators=num_trees, max_features=None, criterion='gini', max_depth=depth)
        clf.fit(Xtrain, ytrain)
        scr = clf.score(Xtest, ytest)
        if scr > test_parse_best:
            test_parse_best = scr
            test_parse_best_tree = num_trees
    test_score.append(test_parse_best)
    if test_parse_best > test_best:
        test_best = test_parse_best
        test_best_depth = depth
        test_best_tree = test_parse_best_tree

plt.xlim(1, 21)
plt.ylim(0, 1.1)
plt.xticks(range(1, 21))
plt.title('5B testing accuracy vs. max_depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.plot(range(1, 21), test_score)
plt.show()

print(train_best)
print(train_best_depth)
print(train_best_tree)
print(test_best)
print(test_best_depth)
print(test_best_tree)
