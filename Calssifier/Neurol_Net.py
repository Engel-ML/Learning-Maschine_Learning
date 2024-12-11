from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Iris-Datensatz laden
iris = load_iris()
X = iris.data
y = iris.target

# 2. Trainings- und Testdatensatz aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y)

# 3. MLP-Classifier instanzieren und trainieren
clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# 4. Vorhersagen auf Testdaten
y_pred = clf.predict(X_test)

# 5. Bewertung
acc = accuracy_score(y_test, y_pred)
print("Genauigkeit:", acc)
print("Konfusionsmatrix:\n", confusion_matrix(y_test, y_pred))
print("Klassifikationsbericht:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. PCA für Visualisierung in 2D
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

y_pred_all = clf.predict(X)

# Farben für jede Klasse definieren (insgesamt 3 Klassen)
colors = ['red', 'green', 'blue']

plt.figure(figsize=(12, 5))
# Subplot 1: Tatsächliche Klassen
plt.subplot(1, 2, 1)
for i, c in enumerate(colors):
    plt.scatter(X_transformed[y==i, 0], X_transformed[y==i, 1], color=c, label=iris.target_names[i])
plt.title("Tatsächliche Klassen (PCA 2D)")
plt.xlabel("Hauptkomponente 1")
plt.ylabel("Hauptkomponente 2")
plt.legend()

# Subplot 2: Vorhergesagte Klassen
plt.subplot(1, 2, 2)
for i, c in enumerate(colors):
    plt.scatter(X_transformed[y_pred_all==i, 0], X_transformed[y_pred_all==i, 1], color=c, label=iris.target_names[i])
plt.title("Vorhergesagte Klassen (PCA 2D)")
plt.xlabel("Hauptkomponente 1")
plt.ylabel("Hauptkomponente 2")
plt.legend()

plt.tight_layout()
plt.show()
