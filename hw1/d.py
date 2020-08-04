import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot

x_ = np.array([0, 5, 8, 12])
y_ = np.array([10, 5, 12, 10])
plot.scatter(x_, y_, color='black')

x = np.array([[1, 0, 0, 0], [1, 5, 25, 125], [1, 8, 64, 512], [1, 12, 144, 1728]])
y = np.array([10, 5, 12, 10])
model = LinearRegression()
model.fit(x, y)

# basic linear fit
x_test = np.array([[x**0, x**1, x**2, x**3] for x in np.linspace(-5, 20, 100)])
y_pred = model.predict(x_test)
#plot.plot(x_test, y_pred, color='red')

# random
x_rand = np.random.uniform(0, 15, 30)
rand_data = np.array([[x**0, x**1, x**2, x**3] for x in x_rand])
y_pred_rand = model.predict(rand_data)
plot.scatter(x_rand, y_pred_rand, color='black')

plot.title("Problem 1(d)")
plot.xlim(-5, 20)
plot.ylim(-5, 20)
plot.show()

