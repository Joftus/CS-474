import numpy as np
import matplotlib.pyplot as plt

x1_1 = np.linspace(0, 100, 1000)
x2_1 = np.linspace(0, -100, 1000)

x1_2 = np.linspace(0, 100, 1000)

x2_3 = np.linspace(0, 100, 1000)

ZERO = np.linspace(0, 0, 1000)

plt.plot(x1_1, x2_1, color='black', label='+1, +2')
plt.plot(x1_2, ZERO, color='gray', label='+1, +3')
plt.plot(ZERO, x2_3, color='blue', label='+2, +3')

plt.text(50, -25, '+1', fontdict={'color': 'black', 'size': 10})
plt.text(15, -50, '+2', fontdict={'color': 'black', 'size': 10})
plt.text(50, 55, '+3', fontdict={'color': 'black', 'size': 10})
# plt.text(50, 55, '+2, +3', fontdict={'color': 'black', 'size': 10})

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

plt.show()
