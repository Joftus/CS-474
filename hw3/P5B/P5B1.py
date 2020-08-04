import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


ytrain = []
Xtrain = []
with open('../train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        if len(row) == 3:
            ytrain.append(int(row[0]))
            Xtrain.append([float(row[1]), float(row[2])])

ytrain = np.array(ytrain)
Xtrain = np.array(Xtrain)

ytest = []
Xtest = []
with open('../test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        if len(row) == 3:
            ytest.append(int(row[0]))
            Xtest.append([float(row[1]), float(row[2])])

Xtest = np.array(Xtest)
ytest = np.array(ytest)

# done with utility

h = .03
x1_min, x1_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
x2_min, x2_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
x1mesh, x2mesh = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

cmap_light = ListedColormap(['lightblue', 'lightcoral', 'grey'])
cmap_bold = ListedColormap(['blue', 'red', 'black'])

for n_neighbors in [1, 5, 15]:
    # we create an instance of Neighbours Classifier and fit the data.

    clf = KNeighborsClassifier(n_neighbors, weights='uniform', algorithm='auto')
    clf.fit(Xtrain, ytrain)

    Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(x1mesh.shape)

    # Plot the training points with the mesh
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytrain_colors = [y - 1 for y in ytrain]
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('%i-NN Training Set' % n_neighbors)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    # Plot the testing points with the mesh
    ypred = clf.predict(Xtest)
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytest_colors = [y - 1 for y in ytest]
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('%i-NN Testing Set' % n_neighbors)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    # Report training and testing accuracies
    print('Working on k=%i' % n_neighbors)
    trainacc = clf.score(Xtrain, ytrain)
    testacc = clf.score(Xtest, ytest)
    print('\tThe training accuracy is %.2f' % trainacc)
    print('\tThe testing accuracy is %.2f' % testacc)
