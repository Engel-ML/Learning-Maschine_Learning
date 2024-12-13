"""
================================
Erkennung handgeschriebener Ziffern
================================

Dieses Beispiel zeigt, wie scikit-learn verwendet werden kann, um Bilder von
handgeschriebenen Ziffern (0-9) zu erkennen.

"""

# Autoren: Die scikit-learn-Entwickler
# Lizenz: BSD-3-Clause

# Standard-Importe für wissenschaftliches Python
import matplotlib.pyplot as plt
#import numpy as np

# Import von Datensätzen, Klassifikatoren und Leistungsmetriken
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




###############################################################################
# Ziffern-Datensatz
# -----------------
#
# Der Ziffern-Datensatz besteht aus 8x8-Pixel-Bildern von Ziffern. 
# Das Attribut ``images`` des Datensatzes speichert 8x8-Arrays von Graustufenwerten
# für jedes Bild. Wir werden diese Arrays verwenden, um die ersten 4 Bilder zu visualisieren.
# Das Attribut ``target`` des Datensatzes speichert die Ziffer, die jedes Bild darstellt, 
# und diese wird im Titel der 4 untenstehenden Diagramme angezeigt.
#
# Hinweis: Wenn wir mit Bilddateien (z. B. 'png'-Dateien) arbeiten würden, 
# würden wir sie mit :func:`matplotlib.pyplot.imread` laden.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Klassifikation
# --------------
#
# Um einen Klassifikator auf diese Daten anzuwenden, müssen wir die Bilder abflachen,
# sodass jedes 2D-Array von Graustufenwerten von der Form ``(8, 8)`` in die Form ``(64,)``
# umgewandelt wird. Anschließend wird der gesamte Datensatz die Form
# ``(n_samples, n_features)`` haben, wobei ``n_samples`` die Anzahl der Bilder 
# und ``n_features`` die Gesamtanzahl der Pixel in jedem Bild ist.
#
# Wir können die Daten dann in Trainings- und Testuntergruppen aufteilen und verschiedene
# Klassifikatoren auf die Trainingsdaten anwenden. Die trainierten Klassifikatoren
# können anschließend verwendet werden, um den Wert der Ziffer für die Proben
# im Testdatensatz vorherzusagen.

# Bilder abflachen
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Verschiedene Klassifikatoren definieren
classifiers = {
    "SVM": svm.SVC(gamma=0.001),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100), #n_estimator gibt an wieviele Entscheidungsbäume generiert werden sollen
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=42),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "QDA": QuadraticDiscriminantAnalysis(),
    "LDA": LinearDiscriminantAnalysis(),
}

# Daten in 50% Trainings- und 50% Testuntergruppen aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Performance-Speicherung
performance_metrics = {}

# Alle Klassifikatoren trainieren und bewerten
for name, clf in classifiers.items():
    print(f"\n{name}")
    # Klassifikator trainieren
    clf.fit(X_train, y_train)
    # Den Wert der Ziffer auf der Testuntergruppe vorhersagen
    predicted = clf.predict(X_test)

    # Klassifikationsbericht ausgeben
    report = metrics.classification_report(y_test, predicted, output_dict=True)
    performance_metrics[name] = report["accuracy"]

    print(
        f"Klassifikationsbericht für {name}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    # Konfusionsmatrix anzeigen
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle(f"Konfusionsmatrix - {name}")

# Visualisierung der Gesamtperformance
plt.figure(figsize=(10, 5))
names = list(performance_metrics.keys())
accuracies = list(performance_metrics.values())
plt.bar(names, accuracies)
plt.title("Genauigkeit der Klassifikatoren")
plt.ylabel("Genauigkeit")
plt.xlabel("Klassifikatoren")
plt.ylim(0, 1)
plt.xticks(rotation=90)
plt.show()
