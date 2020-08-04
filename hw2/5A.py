import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

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
    test_x.append(b[0][2] + b[0][3] + b[0][4] + b[0][5] + b[0][6])
    test_y.append(b[0][0])

plot.scatter(train_x, train_y)
plot.title('HW2train')
plot.xticks(np.arange(min(train_x)-10, max(train_x)+10, 10))
plot.xlabel('X')
plot.ylabel('Y')
plot.show()
