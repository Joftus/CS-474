import numpy as np
import matplotlib.pyplot as plt

circle = plt.Circle((-1, 2), radius=2, facecolor='gray', alpha=0.1, edgecolor='black', linewidth=1.0)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.add_artist(circle)
# plt.text(-1.5, 2, "less than 4", fontdict={'color': 'black', 'size': 10})
# plt.text(-3.5, 4.5, "greater than 4", fontdict={'color': 'black', 'size': 10})
plt.scatter([0, -1, 2, 3], [0, 1, 2, 8], c=['blue', 'red', 'blue', 'blue'])

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Problem 4 A")

plt.show()
