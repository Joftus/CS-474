import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.colors import ListedColormap


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

# done with utility

h = .03
x1_min, x1_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
x2_min, x2_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
x1mesh, x2mesh = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

cmap_light = ListedColormap(['lightblue', 'lightcoral', 'grey'])
cmap_bold = ListedColormap(['blue', 'red', 'black'])


# start of new program
clf = LinearDiscriminantAnalysis(priors=None)
clf.fit(Xtrain, ytrain)
print(clf.score(Xtrain, ytrain))
print(clf.score(Xtest, ytest))

Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])

# Put the result into a color plot
Z = Z.reshape(x1mesh.shape)
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytrain_colors = [y - 1 for y in ytrain]
plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Problem 5 C2 decision regions and training points')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

ypred = clf.predict(Xtest)
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytest_colors = [y - 1 for y in ytest]
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Problem 5 C3 decision regions and testing points')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
