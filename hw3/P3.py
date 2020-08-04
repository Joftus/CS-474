import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
x = np.linspace(-50, 50, 1000)

# Problem 3A
plt.plot(x, (1+3*x), color='black')

# Problem 3B
plt.plot(x, (2-x)/2, color='blue')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Problem 3 A / B')
plt.show()
