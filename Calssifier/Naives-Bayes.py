from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Iris-Datensatz laden
# Der Datensatz besteht aus 150 Instanzen (Blumen), 
# aufgeteilt in 3 Klassen (Setosa, Versicolor, Virginica).
iris = load_iris()
X = iris.data
y = iris.target

# 2. Aufteilen in Trainings- und Testdaten
# Wir nehmen 30% der Daten als Testdaten, 70% als Trainingsdaten.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    stratify=y, 
                                                    random_state=42)

# 3. Naive-Bayes-Klassifikator instanzieren
# GaussianNB ist der gebräuchlichste Naive-Bayes-Klassifikator für kontinuierliche Merkmale,
# da er eine Normalverteilung der Merkmale voraussetzt.
clf = GaussianNB()

# 4. Training des Modells
clf.fit(X_train, y_train)

# 5. Vorhersagen auf den Testdaten treffen
y_pred = clf.predict(X_test)

# 6. Evaluation des Modells
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Genauigkeit:", acc)
print("Konfusionsmatrix:\n", cm)
print("Klassifikationsbericht:\n", report)

# 7. Optionale Visualisierung mit PCA in 2D
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
y_pred_all = clf.predict(X)

colors = ['red', 'green', 'blue']

plt.figure(figsize=(12, 5))

# Tatsächliche Klassen plotten
plt.subplot(1, 2, 1)
for i, c in enumerate(colors):
    plt.scatter(X_transformed[y == i, 0], 
                X_transformed[y == i, 1], 
                color=c, 
                label=iris.target_names[i])
plt.title("Tatsächliche Klassen (PCA 2D)")
plt.xlabel("Hauptkomponente 1")
plt.ylabel("Hauptkomponente 2")
plt.legend()

# Vorhergesagte Klassen plotten
plt.subplot(1, 2, 2)
for i, c in enumerate(colors):
    plt.scatter(X_transformed[y_pred_all == i, 0], 
                X_transformed[y_pred_all == i, 1], 
                color=c, 
                label=iris.target_names[i])
plt.title("Vorhergesagte Klassen (PCA 2D)")
plt.xlabel("Hauptkomponente 1")
plt.ylabel("Hauptkomponente 2")
plt.legend()

plt.tight_layout()
plt.show()
