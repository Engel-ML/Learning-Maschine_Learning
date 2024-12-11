import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Daten laden
iris = load_iris()
X = iris.data[:, [0, 1]]  # Wir nehmen nur die ersten zwei Merkmale
y = iris.target

# KNN-Klassifikator trainieren
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Gitter für die Entscheidungsgrenzen erstellen
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Vorhersagen für jeden Punkt im Gitter
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualisierung
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Sepallänge (cm)')
plt.ylabel('Sepalbreite (cm)')
plt.title('KNN Klassifikation mit k=5')
plt.show()
