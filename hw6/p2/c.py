import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()
# normalize values to [0,1]
X = digits.data / 255


pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
cumsum_var = np.cumsum(pca.explained_variance_ratio_)

# now let's look at some images, reconstructed after only using
# n_components PCA dimensions
X_reproduce = pca.inverse_transform(X_transformed)
for i in range(5):
    plt.matshow(X[i].reshape(8, 8))
    plt.title('Original image sample # = %i' % i)
    plt.matshow(X_reproduce[i].reshape(8, 8))
    plt.title('Reconstructed image sample # = %i' % i)
    plt.show()
