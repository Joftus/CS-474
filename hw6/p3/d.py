import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering #to make scatterplot
from scipy.cluster import hierarchy # to make dendogram

X = []
with open('train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
    for row in reader:
        if len(row) == 3:
            X.append([float(row[1]), float(row[2])])

X = np.array(X)
colors = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00', '#707b7c', '#58d68d', '#1a5276', '#641e16', '#f5cba7', '#212f3d'])

link = 'single'
thresh = 0.39
clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage=link, distance_threshold=thresh, compute_full_tree=True)
clustering.fit(X)

# Plot the training points
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=20, color=colors[clustering.labels_])
plt.title('Scatter plot of the data using %s linkage'% link)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()