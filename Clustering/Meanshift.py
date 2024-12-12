import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# 1. Daten erstellen
centers = [[2, 2], [8, 8], [5, 5]]
X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=1.0, random_state=42)

# 2. Bandbreite schätzen
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)

# 3. MeanShift erstellen und anpassen
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# 4. Visualisierung der Cluster
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Endgültige Zentren')
plt.title("MeanShift Clustering")
plt.legend()
plt.grid()
plt.show()
