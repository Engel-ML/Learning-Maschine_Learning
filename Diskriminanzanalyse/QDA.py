import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 1. Daten laden (Beispiel: Iris-Datensatz)
data = load_iris()
X = data.data[:, :2]  # Nur die ersten zwei Merkmale verwenden
y = data.target

# 2. Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. QDA-Modell erstellen und trainieren
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# 4. Vorhersagen treffen
y_pred = qda.predict(X_test)

# 5. Modellbewertung
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Visualisierung der Entscheidungsgrenzen (nur erste zwei Merkmale)
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Originaldaten (erste zwei Dimensionen)
for class_idx in np.unique(y):
    axs[0].scatter(
        X[y == class_idx, 0],
        X[y == class_idx, 1],
        label=f"Klasse {class_idx}"
    )
axs[0].set_xlabel("Merkmal 1")
axs[0].set_ylabel("Merkmal 2")
axs[0].set_title("Originaldaten (erste zwei Dimensionen)")
axs[0].legend()
axs[0].grid()

# Entscheidungsgrenzen plotten
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axs[1].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
for class_idx in np.unique(y):
    axs[1].scatter(
        X[y == class_idx, 0],
        X[y == class_idx, 1],
        label=f"Klasse {class_idx}"
    )
axs[1].set_xlabel("Merkmal 1")
axs[1].set_ylabel("Merkmal 2")
axs[1].set_title("Entscheidungsgrenzen der QDA (erste zwei Dimensionen)")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
