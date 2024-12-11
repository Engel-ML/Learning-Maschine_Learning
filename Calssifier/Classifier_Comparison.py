# Codequelle: Gaël Varoquaux, Andreas Müller
# Modifiziert für die Dokumentation von Jaques Grobler
# Lizenz: BSD 3-Klausel

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Import verschiedener Klassifikatoren und Hilfsfunktionen aus scikit-learn
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Wir definieren zunächst eine Liste mit den Namen verschiedener Klassifikatoren,
# um diese später auf unsere Datensätze anzuwenden.
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

# Hier werden verschiedene Klassifikatoren instanziiert.
# Jeder Klassifikator ist ein Modelltyp, der für die Klassifikation von Daten genutzt werden kann.
classifiers = [
    KNeighborsClassifier(3),  # k-Nächste-Nachbarn-Klassifikator mit k=3
    SVC(kernel="linear", C=0.025, random_state=42),  # Lineare SVM mit kleinem C-Wert
    SVC(gamma=2, C=1, random_state=42),  # SVM mit RBF-Kern (nicht-linear)
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),  # Gaußscher Prozess
    DecisionTreeClassifier(max_depth=5, random_state=42),  # Entscheidungsbaum mit max. Tiefe 5
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),  # Random Forest aus mehreren Entscheidungsbäumen
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),  # Neuronales Netzwerk (MLP)
    AdaBoostClassifier(algorithm="SAMME", random_state=42),  # AdaBoost-Klassifikator
    GaussianNB(),  # Gaußscher Naive Bayes
    QuadraticDiscriminantAnalysis(),  # Quadratische Diskriminanzanalyse
]

# Wir erzeugen drei verschiedene synthetische Datensätze, die später mit den Klassifikatoren getestet werden.
# 1) Ein linear separabler Datensatz
X, y = make_classification(
    n_features=2,            # Anzahl der Merkmale = 2, damit wir es einfach visualisieren können
    n_redundant=0,           # Keine redundanten Features
    n_informative=2,         # 2 informative Features
    random_state=1,          # Zufallszustand für Reproduzierbarkeit
    n_clusters_per_class=1   # Jede Klasse besteht aus einem Cluster
)

# Wir verschieben die Datenpunkte durch Hinzufügen einer Zufallskomponente, um sie etwas "natürlicher" aussehen zu lassen.
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

# 2) Ein Datensatz in Form von Monden (Halbkreisen)
moons = make_moons(noise=0.3, random_state=0)

# 3) Ein Kreis-Datensatz mit zwei verschachtelten Klassen
circles = make_circles(noise=0.2, factor=0.5, random_state=1)

# Wir packen alle drei Datensätze in eine Liste, um sie später in einer Schleife zu durchlaufen.
datasets = [
    moons,
    circles,
    linearly_separable,
]

# Erstelle eine große Figur mit bestimmten Abmessungen. Hier haben wir drei Datensätze (Zeilen)
# und für jeden Datensatz mehrere Klassifikatoren (Spalten).
figure = plt.figure(figsize=(27, 9))
i = 1  # Zählvariable für die Subplots

# Iteration über die verschiedenen Datensätze
for ds_cnt, ds in enumerate(datasets):
    # ds ist ein Tupel (X, y)
    X, y = ds

    # Aufteilen der Daten in Trainings- (60%) und Testdaten (40%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # Wir bestimmen den Minimal- und Maximalwert der Merkmalsachsen, um später
    # die Plot-Grenzen festzulegen.
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Definiere Farbschemata für die Darstellung
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])  # Rot für Klasse 0, Blau für Klasse 1

    # Erstelle ein Subplot für die Rohdaten (noch ohne Klassifikator)
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")  # Überschrift für die Eingabedaten-Spalte

    # Plot der Trainingspunkte
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot der Testpunkte (mit etwas Transparenz)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")

    # Setze die Achsenlimits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Entferne die Achsenbeschriftungen, um das Diagramm übersichtlicher zu halten
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # Iteration über alle Klassifikatoren
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        # Erstelle eine Pipeline, die zuerst die Daten standardisiert (StandardScaler),
        # dann den Klassifikator anwendet. Das ist oft gute Praxis, vor allem bei SVMs und NN.
        clf = make_pipeline(StandardScaler(), clf)

        # Trainiere den Klassifikator mit den Trainingsdaten
        clf.fit(X_train, y_train)

        # Berechne die Genauigkeit auf den Testdaten als Metrik
        score = clf.score(X_test, y_test)

        # Visualisiere die Entscheidungsgrenzen mit DecisionBoundaryDisplay.from_estimator
        # Dadurch entsteht ein Hintergrund, der zeigt, welcher Klasse welcher Bereich
        # zugeordnet wird.
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plotte erneut die Trainings- und Testpunkte auf die Karte der Entscheidungsgrenzen
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)

        # Setze die gleichen Achsenbereiche wie zuvor
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

        # Die Titelzeile (erste Datenzeile) bekommt die Namen der Klassifikatoren
        if ds_cnt == 0:
            ax.set_title(name)

        # Schreibe den Genauigkeitswert des Klassifikators in das Diagramm
        # "%.2f" formatiert die Zahl auf 2 Nachkommastellen
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),  # lstrip("0") entfernt eine führende Null
            size=15,
            horizontalalignment="right",
        )
        i += 1

# Passe das Layout an, damit nichts überlappt
plt.tight_layout()

# Zeige die komplette Figur an
plt.show()
 