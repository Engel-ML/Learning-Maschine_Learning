import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 1. Daten laden (Beispiel: Iris-Datensatz)
data = load_iris()
X = data.data
y = data.target

# 2. Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. LDA-Modell erstellen und trainieren
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 4. LDA-Transformation der Daten
X_lda = lda.transform(X)
print("Shape der transformierten Daten:", X_lda.shape)

# 5. Vorher-Bild der Originaldaten und Visualisierung der LDA-Richtungen
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

# LDA-Richtungen einzeichnen
mean_overall = np.mean(X, axis=0)
lda_vectors = lda.scalings_[:, :2]  # Erste zwei LD-Richtungen
for i, vec in enumerate(lda_vectors.T):
    axs[0].arrow(mean_overall[0], mean_overall[1], vec[0] * 1, vec[1] * 1, 
                 head_width=0.1, head_length=0.2, fc=f'C{i}', ec=f'C{i}', label=f"LD{i+1}")

# LDA-Projektion (2D-Projektion)
for class_idx in np.unique(y):
    axs[1].scatter(
        X_lda[y == class_idx, 0],
        X_lda[y == class_idx, 1],
        label=f"Klasse {class_idx}"
    )
axs[1].set_xlabel("LD1")
axs[1].set_ylabel("LD2")
axs[1].set_title("LDA-Projektion")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

# 6. Vorhersagen treffen
y_pred = lda.predict(X_test)

# 7. Modellbewertung
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
