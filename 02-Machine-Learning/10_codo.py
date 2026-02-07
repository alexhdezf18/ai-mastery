from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

inercias =[]

for i in range(1, 11):
    modelo = KMeans(n_clusters=i)
    modelo.fit(X)
    inercias.append(modelo.inertia_)

plt.plot(range(1,11), inercias, marker='o')
plt.show()