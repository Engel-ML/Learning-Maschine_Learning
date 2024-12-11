#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:39:36 2024

@author: colinengelmann
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Beispieldaten generieren
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                         n_informative=2, random_state=1, 
                         n_clusters_per_class=1)

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42)

# Decision Tree erstellen und trainieren
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Gitter für Visualisierung erstellen
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Vorhersagen
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualisierung
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title('Decision Tree Klassifikation')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Modell-Performance ausgeben
print(f"Trainingsgenauigkeit: {clf.score(X_train, y_train):.3f}")
print(f"Testgenauigkeit: {clf.score(X_test, y_test):.3f}")