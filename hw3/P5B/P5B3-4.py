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

trainacc = []
testacc = []
best_train = -100
best_test = -100
best_train_k = -100
best_test_k = -100

for n_neighbors in range(1, 31):
    clf = KNeighborsClassifier(n_neighbors, weights='uniform', algorithm='auto')
    clf.fit(Xtrain, ytrain)

    if clf.score(Xtrain, ytrain) > best_train:
        best_train = clf.score(Xtrain, ytrain)
        best_train_k = n_neighbors
    if clf.score(Xtest, ytest) > best_test:
        best_test = clf.score(Xtest, ytest)
        best_test_k = n_neighbors

    trainacc.append(clf.score(Xtrain, ytrain))
    testacc.append(clf.score(Xtest, ytest))

plt.plot(range(1, 31), trainacc)
plt.title('training accuracy')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()

plt.plot(range(1, 31), testacc)
plt.title('testing accuracy')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()

print(best_train)
print(best_train_k)
print(best_test)
print(best_test_k)
