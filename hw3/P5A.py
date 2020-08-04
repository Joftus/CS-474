import numpy as np
import matplotlib.pyplot as plt
import csv

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
for x, y in zip(Xtrain, ytrain):
    if y == 1:
        col = 'blue'
    if y == 2:
        col = 'red'
    if y == 3:
        col = 'black'
    plt.scatter(x[0], x[1], color=col)

plt.title('Scatterplot HW3Train')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

ytest = []
Xtest = []
with open('test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        #         print(row)
        if len(row) == 3:
            ytest.append(int(row[0]))
            Xtest.append([float(row[1]), float(row[2])])

Xtest = np.array(Xtest)
ytest = np.array(ytest)
for x, y in zip(Xtest, ytest):
    #     print(x1,x2,y)
    if y == 1:
        col = 'blue'
    if y == 2:
        col = 'red'
    if y == 3:
        col = 'black'
    plt.scatter(x[0], x[1], color=col)

plt.title('Scatterplot HW3Test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
