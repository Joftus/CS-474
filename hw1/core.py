
# -------| Question 1 |-------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x_orig = np.array([0, 5, 8, 12])
y_orig = np.array([10, 5, 12, 0])

x_pred = np.polyfit(x_orig, y_orig, 3)
model = np.poly1d(x_pred)
y_pred = model(np.linspace(0, 20, 4))

plt.scatter(x_orig, y_orig, s=50, c='black')

# plt.plot(x_orig, y_pred, color='r')
plt.title("Problem 1(b)")
plt.xlim(-5, 20)
plt.ylim(-5, 20)
plt.show()

