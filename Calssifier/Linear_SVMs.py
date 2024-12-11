#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:08:35 2024

@author: colinengelmann
"""

from sklearn import svm
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# Beispieldaten generieren
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                         n_informative=2, random_state=1,
                         n_clusters_per_class=1)

# SVM-Klassifikator erstellen und trainieren
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# Gitter für die Entscheidungsgrenze erstellen
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Vorhersagen für jeden Punkt im Gitter
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualisierung
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM mit linearem Kernel')
plt.show()

# Support Vektoren anzeigen
print(f"Anzahl der Support Vektoren: {len(clf.support_vectors_)}")