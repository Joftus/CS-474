import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


ytrain = [10, 2, 0, 4]
Xtrain = np.array([2, 4, 6, 8]).reshape(-1, 1)
Xtest = np.linspace(-2, 12).reshape(-1, 1)

for n_neighbors in [1, 2, 3, 4]:
    clf = KNeighborsClassifier(n_neighbors, algorithm='auto')
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    plt.title('Problem 3 %i-NN' % n_neighbors)
    plt.scatter(Xtrain, ytrain, color='black')
    plt.plot(Xtest, ypred, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
