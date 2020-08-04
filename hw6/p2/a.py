import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()
print(digits.data.shape)
# normalize values to [0,1]
X = digits.data / 255


pca = PCA(n_components=64)
X_transformed = pca.fit_transform(X)
cumsum_var = np.cumsum(pca.explained_variance_ratio_)

plt.plot(cumsum_var)
plt.title("2a")
plt.ylabel("cumulative explained variance")
plt.show()
