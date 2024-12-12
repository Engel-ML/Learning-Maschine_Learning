#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:51:05 2024

@author: colinengelmann
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Beispiel-Daten erzeugen
np.random.seed(42)
data, labels = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# K-means Clustering ausführen
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(data)
predicted_labels = kmeans.predict(data)

# Cluster-Zentren und Labels visualisieren
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Zentren')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
