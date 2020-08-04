import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KN


train = np.array(pd.read_csv('train.csv', sep='\n'))
train_x = []
train_y = []
test = np.array(pd.read_csv('test.csv', sep='\n'))
test_x = []
test_y = []

for a in train:
    train_x.append(float(a[0][2] + a[0][3] + a[0][4] + a[0][5] + a[0][6]))
    train_y.append(int(a[0][0]))
for b in test:
    test_x.append(float(b[0][2] + b[0][3] + b[0][4] + b[0][5] + b[0][6]))
    test_y.append(int(b[0][0]))


train_x = np.array(train_x).reshape(-1, 1)
train_y = np.array(train_y)
test_x = np.array(test_x).reshape(-1, 1)
test_y = np.array(test_y)

generated_values = np.linspace(0, 100, 1000).reshape(-1, 1)
prediction_x = []
for i in range(0, 1000):
    prediction_x.append(i / 10)

# k = [1, 3, 9]
'''
# Part 1a/b
for c in k:
    model = KN(weights='uniform', algorithm='auto', n_neighbors=c).fit(train_x, train_y)
    prediction = np.array(model.predict(generated_values))
    plot.scatter(train_x, train_y, zorder=2)
    plot.plot(prediction_x, prediction, zorder=1, color='r')
    plot.xticks(np.arange(0, 100, 10))
    plot.xlabel('X')
    plot.ylabel('Y')
    plot.title(str(c) + "nn Classifier with Training data")
    print(str(c) + " score: " + str(model.score(train_x, train_y)))
    plot.show()
'''
'''
# Part 1c/d
for c in k:
    model = KN(weights='uniform', algorithm='auto', n_neighbors=c).fit(test_x, test_y)
    prediction = np.array(model.predict(generated_values))
    plot.scatter(test_x, test_y, zorder=2)
    plot.plot(prediction_x, prediction, zorder=1, color='r')
    plot.xticks(np.arange(0, 100, 10))
    plot.xlabel('X')
    plot.ylabel('Y')
    plot.title(str(c) + "nn Classifier with Testing data")
    print(str(c) + " score: " + str(model.score(test_x, test_y)))
    plot.show()
'''
'''
# Part 2
k = [1, 3, 5, 7, 9, 11, 13, 15]
accuracy = []
for c in k:
    model = KN(weights='uniform', algorithm='auto', n_neighbors=c).fit(train_x, train_y)
    accuracy.append(model.score(train_x, train_y))
plot.plot(k, accuracy)
plot.xlabel('k')
plot.ylabel('accuracy')
plot.title("Training accuracy as a function of k")
plot.show()
'''

# Part 3
k = [1, 3, 5, 7, 9, 11, 13, 15]
accuracy = []
for c in k:
    model = KN(weights='uniform', algorithm='auto', n_neighbors=c).fit(train_x, train_y)
    accuracy.append(model.score(test_x, test_y))
plot.plot(k, accuracy)
plot.xlabel('k')
plot.ylabel('accuracy')
plot.title("Testing accuracy as a function of k")
plot.show()
