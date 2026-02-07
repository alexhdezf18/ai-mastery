from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

modelo = KMeans(n_clusters=4)
modelo.fit(X)

y_kmeans =  modelo.predict(X)
centros = modelo.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centros[:, 0], centros[:, 1], c='black', s=200, alpha=0.5)
plt.show()