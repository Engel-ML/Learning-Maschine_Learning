from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Iris-Datensatz laden
# Der Iris-Datensatz enthält 150 Datenpunkte von Iris-Blumen, aufgeteilt in 3 Klassen:
# - Iris setosa (Klasse 0)
# - Iris versicolor (Klasse 1)
# - Iris virginica (Klasse 2)
# Jede Blume wird durch 4 Merkmale beschrieben (z.B. Kelchblattlänge, -breite, etc.).
iris = load_iris()
X = iris.data
y = iris.target

# 2. Daten in Trainings- und Testdaten aufteilen
# test_size=0.3 bedeutet, dass 30% der Daten für den Test, 70% für das Training verwendet werden.
# stratify=y sorgt dafür, dass alle Klassen im gleichen Verhältnis in Trainings- und Testdaten vorkommen.
# random_state=42 dient der Reproduzierbarkeit.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    stratify=y, 
                                                    random_state=42)

# 3. AdaBoost-Klassifikator instanzieren
# Wir verwenden einen DecisionTreeClassifier mit max_depth=1 als schwachen Lerner.
# n_estimators=50 bedeutet, dass bis zu 50 dieser schwachen Lerner trainiert werden.
# learning_rate=1.0 ist ein Skalierungsfaktor für die Schrittlänge, die bei der Gewichtung der Lerner verwendet wird.
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
clf = AdaBoostClassifier(estimator=base_estimator,
                         n_estimators=50,
                         learning_rate=1.0,
                         algorithm='SAMME',
                         random_state=42)

# 4. Modell an den Trainingsdaten fitten (trainieren)
clf.fit(X_train, y_train)

# 5. Vorhersagen auf den Testdaten treffen
y_pred = clf.predict(X_test)

# 6. Modell bewerten
# accuracy_score gibt die Gesamtgenauigkeit an: Wie viel Prozent der Testbeispiele wurden korrekt klassifiziert?
acc = accuracy_score(y_test, y_pred)

# confusion_matrix zeigt eine tabellarische Übersicht:
# Zeilen: tatsächliche Klassen, Spalten: vorhergesagte Klassen
cm = confusion_matrix(y_test, y_pred)

# classification_report zeigt Precision, Recall und F1-Score für jede Klasse
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Genauigkeit:", acc)
print("Konfusionsmatrix:\n", cm)
print("Klassifikationsbericht:\n", report)

# 7. Optionale Visualisierung
# Wir verwenden PCA, um die vierdimensionalen Daten auf 2 Dimensionen herunterzubrechen.
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

# Vorhersagen auf dem gesamten Datensatz, um eine vollständige Visualisierung zu erhalten
y_pred_all = clf.predict(X)

# Farbzuordnung für jede Klasse (0, 1, 2)
colors = ['red', 'green', 'blue']

plt.figure(figsize=(12, 5))

# Subplot 1: Tatsächliche Klassen
plt.subplot(1, 2, 1)
for i, c in enumerate(colors):
    # Wir plotten nur die Datenpunkte, deren tatsächliche Klasse i ist
    plt.scatter(X_transformed[y == i, 0], 
                X_transformed[y == i, 1], 
                color=c, 
                label=iris.target_names[i])
plt.title("Tatsächliche Klassen (PCA 2D)")
plt.xlabel("Hauptkomponente 1")
plt.ylabel("Hauptkomponente 2")
plt.legend()

# Subplot 2: Vorhergesagte Klassen des AdaBoost-Klassifikators
plt.subplot(1, 2, 2)
for i, c in enumerate(colors):
    # Plotten der Datenpunkte nach ihrer vorhergesagten Klasse
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
