from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Iris Datensatz laden
iris = load_iris()
X = iris.data
y = iris.target

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

# Random Forest erstellen und trainieren
rf = RandomForestClassifier(
    n_estimators=100,      # Anzahl der Bäume
    max_depth=3,           # Maximale Tiefe der Bäume
    random_state=42
)

# Model trainieren
rf.fit(X_train, y_train)

# Feature Importance visualisieren
importances = rf.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(10, 6))
plt.title('Wichtigkeit der Iris-Merkmale')
plt.bar(feature_names, importances)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vorhersagen für zwei Merkmale visualisieren
plt.figure(figsize=(10, 6))
# Wir nutzen nur Länge und Breite der Blütenblätter
X = iris.data[:, [2, 3]]
y = iris.target

# Neues Modell mit nur 2 Features trainieren
rf_2d = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf_2d.fit(X, y)

# Gitter für Visualisierung erstellen
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Vorhersagen
Z = rf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Blütenblattlänge')
plt.ylabel('Blütenblattbreite')
plt.title('Random Forest Klassifikation der Iris-Arten')
plt.show()

# Genauigkeit ausgeben
print(f'Genauigkeit auf Testdaten: {rf.score(X_test, y_test):.3f}')