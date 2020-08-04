import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LogisticRegression as LR

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

# model = LR(penalty='none', fit_intercept=True).fit(train_x, train_y)
model = LR(penalty='none', fit_intercept=True).fit(test_x, test_y)
generated_values = np.linspace(0, 100, 1000).reshape(-1, 1)
probability = []
for i in range(0, 1000):
    probability.append(model.predict_proba(generated_values)[i][1])
probability_x = []
for i in range(0, 1000):
    probability_x.append(i / 10)

'''
# Parts 1, 2
print(model.intercept_)
print(model.coef_)
print(model.score(train_x, train_y))
'''
'''
# Part 3
plot.scatter(train_x, train_y, zorder=2)
plot.plot(probability_x, probability, color='r', zorder=1)
plot.title('HW2train Scatter Plot and Prob(y = 1|x)')
plot.xticks(np.arange(0, 100, 10))
plot.xlabel('X')
plot.ylabel('Y')
plot.show()
'''
'''
# Part 4
plot.scatter(test_x, test_y, zorder=2)
plot.plot(probability_x, probability, color='r', zorder=1)
plot.title('HW2test Scatter Plot and Prob(y = 1|x)')
plot.xticks(np.arange(0, 100, 10))
plot.xlabel('X')
plot.ylabel('Y')
plot.show()
'''

# Part 5
# print(model.score(test_x, test_y))
