import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()
# normalize values to [0,1]
X = digits.data / 255


pca = PCA(n_components=64)
X_transformed = pca.fit_transform(X)
cumsum_var = np.cumsum(pca.explained_variance_ratio_)

first = -1
second = -1
cumexp_var = pca.explained_variance_

for cumexp in cumsum_var:
    if cumexp >= second:
        if cumexp >= first:
            second = first
            first = cumexp
        else:
            second = cumexp

print(first)
print(second)
