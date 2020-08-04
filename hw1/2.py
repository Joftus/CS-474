import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
import itertools

# -------| Question 2 |-------
import pandas as pd
# range = 506

data = pd.read_csv('housingdata.csv')
crim = np.array(data['CRIM'])
zn = np.array(data['ZN'])
indus = np.array(data['INDUS'])
chas = np.array(data['CHAS'])
nox = np.array(data['NOX'])
rm = np.array(data['RM'])
age = np.array(data['AGE'])
dis = np.array(data['DIS'])
rad = np.array(data['RAD'])
tax = np.array(data['TAX'])
ptratio = np.array(data['PTRATIO'])
b = np.array(data['B'])
lstat = np.array(data['LSTAT'])
medv = np.array(data['MEDV'])


features = [
    crim, zn, indus, chas, nox, rm,
    age, dis, rad, tax, ptratio, b, lstat
]

feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
    'TAX', 'PTRATIO', 'B', 'LSTAT'
]
'''
index = 0
for feat in features:
    x_axis = feat[:400]
    y_axis = medv[:400]
    x_pred = np.polyfit(x_axis, y_axis, 1)
    model = np.poly1d(x_pred)
    y_pred = model(x_axis)

    # plot.scatter(x_axis, y_axis)
    plot.plot(x_axis, y_pred, color='r')
    plot.title('{0} fitted linear model'.format(feature_names[index]))
    plot.xlabel(feature_names[index])
    plot.ylabel('MEDV')
    plot.show()
    index += 1
'''

def core(lst, n):
    data = []
    if n == 1:
        data = lst
    if n == 2:
        for i in range(0, len(lst[0])):
            data.append([lst[0][i], lst[1][i]])
    if n == 3:
        for i in range(0, len(lst[0])):
            data.append([lst[0][i], lst[1][i], lst[2][i]])
    if n == 4:
        for i in range(0, len(lst[0])):
            data.append([lst[0][i], lst[1][i], lst[2][i], lst[3][i]])

    if n == 5:
        for i in range(0, len(lst[0])):
            data.append([lst[0][i], lst[1][i], lst[2][i], lst[3][i], lst[4][i]])
    model = LinearRegression()
    model.fit(data, medv)
    



X = [age, indus, nox, rm, tax]
feature_names_2 = ['AGE', 'INDUS', 'NOX', 'RM', 'TAX']
for combo in itertools.combinations(feature_names_2, len(X)):


# [-106:]
# [:400]