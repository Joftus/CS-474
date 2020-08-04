import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
# This code will suppress warnings.
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
# load training set
ytrain = []
Xtrain = []
with open('train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        if len(row) == 3:
            ytrain.append(int(row[0]))
            Xtrain.append([float(row[1]), float(row[2])])

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
# load testing set
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
# train network with a single hidden layer of 2 nodes

clf = MLPClassifier(hidden_layer_sizes=(2,), activation='tanh', solver='sgd', max_iter=1)

clf.fit(Xtrain, ytrain)
print(clf.coefs_)
print('\n\n')
print(clf.intercepts_)

w11 = np.linspace(-10, 10, 75)
w21 = np.linspace(-10, 10, 75)

# create a meshgrid and evaluate training MSE
W11, W21 = np.meshgrid(w11, w21)
MSEmesh = []
for coef1, coef2 in np.c_[W11.ravel(), W21.ravel()]:
    clf.coefs_[1][1][1] = coef2
    clf.coefs_[0][0][0] = coef1
    MSEmesh.append([clf.score(Xtrain, ytrain)])

MSEmesh = np.array(MSEmesh)

# Put the result into a color plot
MSEmesh = MSEmesh.reshape(W11.shape)

ax = plt.axes(projection='3d')
ax.plot_surface(W11, W21, MSEmesh, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Training MSE')
plt.show()
