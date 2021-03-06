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

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)

depths = [1, 2, 3, 4, 10]

for depth in depths:
    c_val = -1
    best_train = -1
    best_test = -1

    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth= depth)
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
    plt.title('4A max depth: %d' % depth)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

