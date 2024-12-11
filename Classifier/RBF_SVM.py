from sklearn import svm
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Iris-Daten laden
iris = load_iris()
X = iris.data[:, [0, 1]]  # Erste zwei Merkmale: Sepallänge und Sepalbreite
y = iris.target

# SVM-Klassifikator mit RBF-Kernel erstellen und trainieren
clf = svm.SVC(kernel='rbf', gamma='scale', C=1.0)
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
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel('Sepallänge (cm)')
plt.ylabel('Sepalbreite (cm)')
plt.title('SVM mit RBF-Kernel auf Iris-Daten')
plt.colorbar()
plt.show()

# Support Vektoren markieren
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=100, linewidth=1, facecolors='none', edgecolors='k')