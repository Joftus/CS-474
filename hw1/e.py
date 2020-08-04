import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot

x_axis = np.sort(np.random.uniform(-10, 20, 100))


x = np.array([[1, 0, 0, 0], [1, 5, 25, 125], [1, 8, 64, 512], [1, 12, 144, 1728]])
y = np.array([10, 5, 12, 10])
model = LinearRegression()
model.fit(x, y)

data = np.array([[x**0, x**1, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14, x**15,
                  x**16, x**17, x**18, x**19, x**20, x**21, x**22, x**23, x**24, x**25, x**26, x**27, x**28, x**29, x**30] for x in np.linspace(-5, 20, 100)])
x_test = np.array([[x**0, x**1, x**2, x**3] for x in np.linspace(-5, 20, 100)])
y_pred = model.predict(data)

mse = []
mae = []
stlf = []
y_axis_lst = [mse, mae, stlf]
y_axis_name = ['MSE', 'MAE', 'STLF']

for i in range(0, len(x_axis)):
    res = x_axis[i] - y_pred[i]
    mse.append((1/100) * abs(res * res))
    mae.append((1/100) * abs(res))
    if mae[i] < 0:
        mae *= -1
    if res < 0:
        res *= -.2
    else:
        res *= 10
    stlf.append(res * (1 / (abs(x_test[i]-5) + .01)))

for a in range(0, 3):
    plot.plot(x_axis, y_axis_lst[a], color='black')
    plot.title(y_axis_name[a])
    plot.xlabel("a_0")
    plot.ylabel(y_axis_name[a])
    plot.show()
