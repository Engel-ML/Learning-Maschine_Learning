# Maschine Learning Notizen

- Supervised Learning
    - **Classifier**
        - K-Nearest Neighboors
            
            KNN ist eine nichtparametrische Methode zur Schätzung von Wahrscheinlichkeitsdichtefunktionen. Der daraus resultierende K-nearest-Neighboor-Algorithmus ist ein Klassifikationsverfahren, bei dem eine Klassenzuordnung unter Berücksichtigung seiner k nächsten Nachbarn vorgenommen wird 
            
            Die mathematische Funktionsweise des kNN-Algorithmus basiert auf Distanzberechnungen:
            
            1. Für einen neuen Datenpunkt x werden die Abstände zu allen Trainingsdatenpunkten berechnet. Die häufigste Metrik ist die euklidische Distanz:
            
            ```python
            d(x,y) = √(Σ(xi - yi)²)
            ```
            
            1. Die k nächsten Nachbarn (kleinste Distanzen) werden identifiziert
            2. Die Klassenzuordnung erfolgt durch Mehrheitsentscheidung: Die Klasse, die unter den k nächsten Nachbarn am häufigsten vorkommt, wird dem neuen Datenpunkt zugewiesen
            
            Optional kann auch eine gewichtete Abstimmung verwendet werden, bei der nähere Nachbarn stärker gewichtet werden als entferntere:
            
            ```python
            w(i) = 1/d(x,xi)²
            ```
            
            Die Wahl des Parameters k ist entscheidend: Ein zu kleines k macht den Algorithmus anfällig für Rauschen, ein zu großes k verwischt die Klassengrenzen.
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image.png)
            
            # KNN Visualisierung
            
            Hier ist ein Code-Beispiel, das die Iris-Daten visualisiert und die KNN-Klassifikationsgrenzen zeigt:
            
            ```python
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
            
            ```
            
            Dieser Code erstellt eine Visualisierung, die:
            
            - Die Datenpunkte als farbige Punkte darstellt (jede Farbe repräsentiert eine Klasse)
            - Die Entscheidungsgrenzen des KNN-Klassifikators als farbige Regionen zeigt
            - Die ersten beiden Merkmale der Iris-Daten (Sepallänge und Sepalbreite) verwendet
            
            ```mermaid
            graph TD;
                A["Input: Neuer Datenpunkt"] --> B["Distanzberechnung"];
                B --> C["Berechne euklidische Distanz zu allen Trainingspunkten"];
                C --> D["Sortiere Distanzen"];
                D --> E["Wähle k nächste Nachbarn"];
                E --> F["Mehrheitsentscheidung"];
                F --> G["Klassenzuweisung"];
                
                %% Optionale Gewichtung
                E --> H["Optional: Gewichtete Abstimmung"];
                H --> F;
                
                %% Details zur Distanzberechnung
                C --> I["d(x,y) = √(Σ(xi - yi)²)"];
                
                %% Parameter k
                J["Wahl von k"] --> E;
                J --> K["k zu klein: Anfällig für Rauschen"];
                J --> L["k zu groß: Verwischte Klassengrenzen"];
            ```
            
            - Support Vector Machine
                - Lineare SVMs
                    
                    Die Support Vector Machine (SVM) ist ein leistungsfähiger Klassifikator und Regressor. Das Grundprinzip besteht darin, Objekte in Klassen zu unterteilen, wobei um die Klassengrenzen ein möglichst breiter Bereich (Margin) frei von Objekten bleibt.
                    
                    Der Algorithmus basiert auf folgenden mathematischen Grundlagen:
                    
                    - Linearer Fall:
                        - Die Trennebene wird durch w^T x + b = 0 beschrieben
                        - w ist der Normalenvektor der Hyperebene
                        - b ist der Bias-Term
                    
                    Die Optimierungsfunktion lautet:
                    
                    ```python
                    minimize: (1/2)||w||²
                    subject to: yi(w^T xi + b) ≥ 1 für alle i
                    ```
                    
                    Dabei gilt:
                    
                    - yi sind die Klassenlabels (+1 oder -1)
                    - xi sind die Eingabevektoren
                    - Die Nebenbedingungen stellen sicher, dass alle Punkte korrekt klassifiziert werden
                    
                    In der Praxis wird oft die duale Form verwendet:
                    
                    ```python
                    maximize: Σαi - (1/2)ΣΣαiαjyiyj(xi^T xj)
                    subject to: αi ≥ 0 und Σαiyi = 0
                    ```
                    
                    Für nicht-linear trennbare Daten wird der Kernel-Trick verwendet:
                    
                    - Die Daten werden in einen höherdimensionalen Raum transformiert
                    - Häufige Kernel-Funktionen sind:
                        - Polynomial: K(x,y) = (x^T y + c)^d
                        - RBF: K(x,y) = exp(-γ||x-y||²)
                    
                    Die Implementierung erfolgt meist mit Soft-Margin-Klassifikation, die einen Parameter C einführt:
                    
                    - C kontrolliert den Trade-off zwischen Margin-Maximierung und Trainingsfehler
                    - Größeres C bedeutet weniger Toleranz für Fehlklassifikationen
                    - Kleineres C erlaubt mehr Fehler zugunsten einer breiteren Margin
                    
                    Die SVM hat mehrere Vorteile:
                    
                    - Effektiv in hochdimensionalen Räumen
                    - Speichereffizient durch Verwendung von Support Vektoren
                    - Vielseitig durch verschiedene Kernel-Funktionen
                    
                    Hier ist ein Beispiel für die Implementierung einer SVM mit scikit-learn:
                    
                    ```python
                    from sklearn import svm
                    from sklearn.datasets import make_classification
                    import numpy as np
                    import matplotlib.pyplot as plt
                    
                    # Beispieldaten generieren
                    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                             n_informative=2, random_state=1,
                                             n_clusters_per_class=1)
                    
                    # SVM-Klassifikator erstellen und trainieren
                    clf = svm.SVC(kernel='linear', C=1.0)
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
                    plt.figure(figsize=(8, 6))
                    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
                    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
                    plt.xlabel('Feature 1')
                    plt.ylabel('Feature 2')
                    plt.title('SVM mit linearem Kernel')
                    plt.show()
                    
                    # Support Vektoren anzeigen
                    print(f"Anzahl der Support Vektoren: {len(clf.support_vectors_)}")
                    ```
                    
                    Dieser Code:
                    
                    - Generiert synthetische Klassifikationsdaten
                    - Erstellt und trainiert einen SVM-Klassifikator mit linearem Kernel
                    - Visualisiert die Entscheidungsgrenze und Datenpunkte
                    - Zeigt die Anzahl der verwendeten Support Vektoren an
                    
                    ```mermaid
                    graph TD;
                        A["Input: Trainingsdaten"] --> B["Initialisierung"];
                        B --> C["Optimierungsproblem aufstellen"];
                        
                        C --> D["Minimiere: (1/2)||w||²"];
                        D --> E["Unter Nebenbedingungen:"];
                        E --> F["yi(w^T xi + b) ≥ 1"];
                        
                        %% Duales Problem
                        C --> G["Duales Problem"];
                        G --> H["Lagrange-Multiplikatoren αi"];
                        H --> I["Maximiere: Σαi - (1/2)ΣΣαiαjyiyj(xi^T xj)"];
                        
                        %% Support Vektoren
                        I --> J["Support Vektoren identifizieren"];
                        J --> K["Punkte mit αi > 0"];
                        
                        %% Hyperebene
                        K --> L["Hyperebene berechnen"];
                        L --> M["w = Σαiyi xi"];
                        L --> N["b = yi - w^T xi"];
                        
                        %% Klassifikation
                        M --> O["Klassifikationsfunktion"];
                        N --> O;
                        O --> P["f(x) = sign(w^T x + b)"];
                        
                        %% Ergebnis
                        P --> Q["Output: Optimale Trennebene"]
                    ```
                    
                - RBF SVM
                    
                    Der RBF-Kernel ist eine der am häufigsten verwendeten Kernel-Funktionen für SVMs. Im Gegensatz zum linearen Kernel kann der RBF-Kernel nicht-lineare Entscheidungsgrenzen erzeugen.
                    
                    Mathematische Definition:
                    
                    ```python
                    K(x,y) = exp(-γ||x-y||²)
                    
                    ```
                    
                    Dabei gilt:
                    
                    - γ (gamma) ist ein Hyperparameter, der die Breite des Kernels kontrolliert
                    - ||x-y||² ist die quadrierte euklidische Distanz zwischen zwei Datenpunkten
                    
                    Unterschiede zum linearen Kernel:
                    
                    - Der RBF-Kernel kann nicht-lineare Beziehungen in den Daten modellieren, während der lineare Kernel nur lineare Trennungen ermöglicht
                    - RBF hat einen zusätzlichen Parameter γ, der optimiert werden muss
                    - RBF transformiert die Daten implizit in einen unendlich-dimensionalen Raum
                    
                    Hier ist ein Code-Beispiel mit dem RBF-Kernel:
                    
                    ```python
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
                    ```
                    
                    Wie beim linearen Kernel kontrolliert der Parameter C den Trade-off zwischen Margin-Maximierung und Trainingsfehler. Ein größeres C bedeutet weniger Toleranz für Fehlklassifikationen, während ein kleineres C mehr Fehler zugunsten einer breiteren Margin erlaubt.
                    
                - Vergleich
                    
                    Der folgende Code nutzt verschiedene Kernels bei den SVMs. Darunter einen linearen, RBF und 
                    
                    Dieser Code demonstriert die Anwendung eines GaussianProcessClassifiers (Gaußprozess-Klassifikators) auf den Iris-Datensatz unter Verwendung zweier verschiedener Kernel-Konfigurationen – einer isotropen und einer anisotropen RBF-Kernelvariante – und visualisiert die resultierenden Klassifikationsgrenzen sowie die Wahrscheinlichkeiten für jede Klasse im zweidimensionalen Merkmalsraum.
                    
                    **Was passiert Schritt für Schritt?**
                    
                    1. **Daten laden und Vorverarbeiten:**
                        - Mit `iris = datasets.load_iris()` wird der Iris-Datensatz geladen, der Informationen zu drei Iris-Blumenarten enthält.
                        - `X = iris.data[:, :2]` wählt nur die ersten beiden Merkmale aus, hier sind das meist `Sepal length` und `Sepal width` (Kelchblattlänge und -breite). Dadurch können wir die Daten in einem zweidimensionalen Plot darstellen.
                        - `y = np.array(iris.target, dtype=int)` liefert die zugehörigen Klassenlabels als ganze Zahlen (0, 1, 2) für die drei Iris-Arten.
                    2. **Definition der Kernel und Modelltraining:**
                        - Es werden zwei Gaußprozess-Klassifikatoren aufgestellt.
                            - Der erste `gpc_rbf_isotropic` verwendet einen isotropen RBF-Kernel. "Isotrop" bedeutet, dass der Kernel in allen Richtungen im Merkmalsraum gleich "schnell" abfällt. Hier wird der Kernel mit `RBF([1.0])` definiert, was einen einzelnen Längenskalenparameter verwendet.
                            - Der zweite `gpc_rbf_anisotropic` verwendet einen anisotropen RBF-Kernel mit `RBF([1.0, 1.0])`. Das bedeutet, dass der Kernel für jede Richtung im Merkmalsraum (also für jede Eingangsvariable) eine eigene Längenskala hat. So kann der Kernel in einer Merkmalsdimension anders "skaliert" sein als in der anderen.
                        
                        Beide Modelle werden anschließend mit `fit(X, y)` an den Daten trainiert.
                        
                    3. **Erzeugen eines Rasters (Mesh) zur Visualisierung:**
                        - Mit `np.meshgrid` wird ein Gitter von Punkten über den Bereich der betrachteten Merkmalswerte erstellt. Dieses Gitter deckt den gesamten Merkmalsraum ab, in dem unsere Trainingspunkte liegen (leicht vergrößert, damit man die Klassifikationsgrenzen schön sieht).
                        - Das Raster dient dazu, die Vorhersage des Modells an vielen Punkten abzurufen, um anschließend die Entscheidungsgrafik (Karten der Klassenwahrscheinlichkeiten) plotten zu können.
                    4. **Vorhersage und Wahrscheinlichkeitsdarstellung:**
                        - Für alle Punkte im Mesh wird `clf.predict_proba(...)` aufgerufen, um die Wahrscheinlichkeiten für jede Klasse vorherzusagen.
                        - Die Vorhersagen werden anschließend zu einer Form umgewandelt, mit der `imshow` umgehen kann. So entsteht eine farbige Karte, auf der jeder Punkt im Merkmalsraum eine Farbgebung entsprechend der Klassenwahrscheinlichkeiten erhält.
                    5. **Plotten der Ergebnisse:**
                        - Der Code erzeugt zwei nebeneinander liegende Plots (`plt.subplot(1, 2, ...)`), einen für den isotropen und einen für den anisotropen Kernel.
                        - `plt.imshow()` zeigt die Wahrscheinlichkeitsverteilung der Klassen im Hintergrund an.
                        - Mit `plt.scatter()` werden die Trainingspunkte (unsere Iris-Blumen), eingefärbt nach ihrer wahren Klasse (rot, grün, blau), in den Plot eingefügt.
                        - Die Achsen werden entsprechend bezeichnet (`Sepal length`, `Sepal width`), und die Titel enthalten den Namen des Kernels sowie den Wert der "Log Marginal Likelihood" (LML). Die LML ist ein Indikator, wie gut das Modell zu den Daten passt.
                    6. **Vergleich der beiden Modelle:**
                        - Durch den Vergleich der beiden Subplots sieht man, wie unterschiedlich isotrope und anisotrope RBF-Kernel den Raum "krümmen" und wie sich dadurch die Entscheidungsgrafiken unterscheiden.
                        - Das anisotrope Modell kann in Richtung eines Merkmals stärker oder schwächer variieren als in Richtung des anderen, was potenziell zu besseren Anpassungen führt, falls die Daten in einer Dimension sensibler auf Veränderungen reagieren als in der anderen.
                    
                    **Kurz gesagt:**
                    
                    - Der Code führt eine Gaußprozess-Klassifikation auf den Iris-Daten durch.
                    - Er verwendet zwei verschiedene Kernel, um die Auswirkung auf die Klassifikationslandschaft zu zeigen.
                    - Das Ergebnis ist eine visuelle Darstellung der Wahrscheinlichkeiten für jede Klasse im Merkmalsraum, zusammen mit den Trainingsdatenpunkten und einem Vergleich der Modellgüte über die log-marginalen Likelihoods.
                    - Code
                        
                        ```bash
                        import matplotlib.pyplot as plt
                        import numpy as np
                        from sklearn import datasets, svm
                        
                        iris = datasets.load_iris()
                        X = iris.data
                        y = iris.target
                        
                        # Für dieses Beispiel benutzen wir nur zwei der Klassen (1 und 2) und zwei Features
                        X = X[y != 0, :2]
                        y = y[y != 0]
                        
                        n_sample = len(X)
                        
                        np.random.seed(17)
                        order = np.random.permutation(n_sample)
                        X = X[order]
                        y = y[order].astype(float)
                        
                        X_train = X[: int(0.9 * n_sample)]
                        y_train = y[: int(0.9 * n_sample)]
                        X_test = X[int(0.9 * n_sample) :]
                        y_test = y[int(0.9 * n_sample) :]
                        
                        # fit the model
                        for kernel in ("linear", "rbf", "poly"):
                            clf = svm.SVC(kernel=kernel, gamma=10)
                            clf.fit(X_train, y_train)
                            
                            # Berechne die Testgenauigkeit
                            accuracy = clf.score(X_test, y_test)
                        
                            plt.figure()
                            plt.clf()
                            plt.scatter(
                                X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolor="k", s=20
                            )
                        
                            # Markiere die Testdaten
                            plt.scatter(
                                X_test[:, 0], X_test[:, 1], s=80, facecolors="none", zorder=10, edgecolor="k"
                            )
                        
                            plt.axis("tight")
                            x_min = X[:, 0].min()
                            x_max = X[:, 0].max()
                            y_min = X[:, 1].min()
                            y_max = X[:, 1].max()
                        
                            XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
                            Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
                        
                            # Put the result into a color plot
                            Z = Z.reshape(XX.shape)
                            plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired, shading='auto')
                            plt.contour(
                                XX,
                                YY,
                                Z,
                                colors=["k", "k", "k"],
                                linestyles=["--", "-", "--"],
                                levels=[-0.5, 0, 0.5],
                            )
                            
                            # Titel mit Classifier-Name und Genauigkeit
                            plt.title(f"{kernel} Kernel (Accuracy: {accuracy:.2f})")
                        
                        plt.show()
                        
                        ```
                        
                    
                    ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%201.png)
                    
                    ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%202.png)
                    
                    ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%203.png)
                    
        - Gauß-Prozess Classifier
            
            Der Gauß-Prozess Klassifikator (GPC) ist ein probabilistischer Klassifikationsalgorithmus, der auf der Theorie der Gauß-Prozesse basiert. Er bietet nicht nur Klassifikationsentscheidungen, sondern auch Wahrscheinlichkeitsschätzungen für die Vorhersagen.
            
            ## Mathematische Grundlagen
            
            Ein Gauß-Prozess ist eine Sammlung von Zufallsvariablen, wobei jede endliche Teilmenge eine multivariate Normalverteilung besitzt. Er wird vollständig durch seine Mittelwertfunktion m(x) und Kovarianzfunktion k(x,x') definiert:
            
            $$
            f(x) ~ GP(m(x), k(x,x'))
            $$
            
            Die Kovarianzfunktion (auch Kernel genannt) definiert die Ähnlichkeit zwischen Datenpunkten. Eine häufig verwendete Kovarianzfunktion ist der RBF-Kernel:
            
            $$
            k(x,x') = σ² exp(-\frac{||x-x'||²}{2l²})
            $$
            
            Für die Klassifikation wird der Gauß-Prozess mit einer Sigmoid-Funktion (meist probit oder logistic) kombiniert, um Wahrscheinlichkeiten zu erhalten:
            
            $$
            P(y=1|x) = σ(f(x))
            $$
            
            ## Implementierung in Python
            
            ```python
            import numpy as np
            from sklearn.datasets import make_classification
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import RBF
            import matplotlib.pyplot as plt
            
            # Beispieldaten generieren
            X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                     n_informative=2, random_state=1,
                                     n_clusters_per_class=1)
            
            # Gauß-Prozess Klassifikator erstellen
            kernel = 1.0 * RBF([1.0])
            gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
            
            # Model trainieren
            gpc.fit(X, y)
            
            # Gitter für Visualisierung erstellen
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
            
            # Vorhersagen und Wahrscheinlichkeiten
            Z = gpc.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            prob = gpc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            prob = prob.reshape(xx.shape)
            
            # Visualisierung
            plt.figure(figsize=(10, 5))
            
            # Entscheidungsgrenzen
            plt.subplot(121)
            plt.contourf(xx, yy, Z, alpha=0.4)
            plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
            plt.title('Klassifikationsgrenzen')
            
            # Wahrscheinlichkeiten
            plt.subplot(122)
            plt.contourf(xx, yy, prob, alpha=0.4)
            plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
            plt.title('Wahrscheinlichkeitsverteilung')
            
            plt.show()
            ```
            
            | Vorteile | Nachteile |
            | --- | --- |
            | Probabilistische Vorhersagen: Liefert Wahrscheinlichkeiten statt nur harter Klassifikationen | Rechenaufwand: O(n³) Komplexität für n Trainingspunkte |
            | Automatische Modellkomplexität: Passt sich automatisch an die Komplexität der Daten an | Speicherbedarf: Benötigt O(n²) Speicher |
            | Unsicherheitsschätzung: Kann die Unsicherheit seiner Vorhersagen quantifizieren | Hyperparameter-Optimierung: Kernel-Parameter müssen sorgfältig gewählt werden |
        - Decision Tree
            
            Decision Tree ist ein überwachter Lernalgorithmus ohne Parameter, der sowohl für Klassifizierungs- als auch Regressionsaufgaben verwendet wird. Er hat eine hierarchische Baumstruktur, die aus einem Stammknoten, Zweigen, inneren Knoten und Blattknoten besteht.
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%204.png)
            
            Wie man dem Diagramm entnehmen kann, beginnt der Decision Tree mit einem Stammknoten, der keine eingehenden Zweige hat. Die vom Stammknoten ausgehenden Zweige münden dann in die inneren Knoten, auch Entscheidungsknoten genannt. Basierend auf den verfügbaren Funktionen führen beide Knotentypen Auswertungen durch, um homogene Teilmengen zu bilden, die als Blattknoten oder Endknoten bezeichnet werden. Die Blattknoten stellen alle möglichen Ergebnisse innerhalb des Datensatzes dar.
            
            ## Mathematische Grundlagen
            
            Die wichtigsten mathematischen Metriken für Decision Trees sind:
            
            1. Gini-Index:
            
            $$
            Gini = 1 - Σ(pi)²
            $$
            
            wobei pi die Wahrscheinlichkeit der Klasse i ist.
            
            1. Entropie:
            
            $$
            Entropy = -Σ(pi * log2(pi))
            $$
            
            1. Informationsgewinn:
            
            $$
            IG(T,a) = H(T) - Σ((|Tv|/|T|) * H(Tv))
            $$
            
            wobei H(T) die Entropie des Elternknotens und H(Tv) die Entropie der Kindknoten ist.
            
            ## Implementierung in Python
            
            ```python
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Beispieldaten generieren
            X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                     n_informative=2, random_state=1, 
                                     n_clusters_per_class=1)
            
            # Daten aufteilen
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                                random_state=42)
            
            # Decision Tree erstellen und trainieren
            clf = DecisionTreeClassifier(max_depth=3, random_state=42)
            clf.fit(X_train, y_train)
            
            # Gitter für Visualisierung erstellen
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
            
            # Vorhersagen
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Visualisierung
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, alpha=0.4)
            plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
            plt.title('Decision Tree Klassifikation')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()
            
            # Modell-Performance ausgeben
            print(f"Trainingsgenauigkeit: {clf.score(X_train, y_train):.3f}")
            print(f"Testgenauigkeit: {clf.score(X_test, y_test):.3f}")
            ```
            
            Dieser Code:
            
            - Erstellt einen Decision Tree Classifier mit maximaler Tiefe 3
            - Trainiert das Modell mit synthetischen Klassifikationsdaten
            - Visualisiert die Entscheidungsgrenzen
            - Berechnet und zeigt die Modellgenauigkeit auf Trainings- und Testdaten
            
            Der **Decision Tree** wählt seine Abzweigungen basierend auf der Metrik aus, die die beste Trennung der Daten ermöglicht. Bei jeder Verzweigung:
            
            Werden alle verfügbaren Features und möglichen Splitpunkte evaluiert
            
            Für jeden möglichen Split wird die gewählte Metrik (Gini-Index oder Entropie) berechnet
            
            Der Split mit dem höchsten Informationsgewinn wird ausgewählt
            
            Beispiel für die Gini-Berechnung:
            
            Ein Knoten mit gleicher Anzahl von zwei Klassen hat einen Gini von 0.5
            
            Ein reiner Knoten (nur eine Klasse) hat einen Gini von 0
            
            Je niedriger der Gini-Index, desto besser die Trennung
            
            Der Prozess wird rekursiv fortgesetzt, bis eines der Abbruchkriterien erreicht ist (z.B. maximale Tiefe oder minimale Anzahl von Samples pro Knoten).
            
            | Aspekt | Vorteile | Nachteile |
            | --- | --- | --- |
            | Interpretation | Leicht verständlich und visualisierbar | Kann bei komplexen Daten unübersichtlich werden |
            | Leistung | Schnelles Training und Vorhersagen | Oft weniger genau als komplexere Modelle |
            | Datenbedarf | Funktioniert auch mit kleinen Datensätzen | Benötigt saubere, gut strukturierte Daten |
            | Robustheit | Keine Annahmen über Datenverteilung nötig | Anfällig für Überanpassung (Overfitting) |
            | Flexibilität | Geeignet für numerische und kategorische Daten | Schwierig bei kontinuierlichen numerischen Ausgaben |
            
        - Random Forest
            
            Zu Deutsch "Zufallswald", besteht aus mehreren Decision Trees. Wie zuvor beschrieben, versuchen Decision Trees die beste Aufteilung der Daten zu finden. Sie können allerdings anfällig für Probleme wie Verzerrung und Überanpassung sein. Wenn im Random-Forest-Algorithmus mehrere Decision Trees ein Ensemble bilden, liefern sie genauere Ergebnisse, insbesondere wenn die einzelnen Bäume nicht miteinander korrelieren.
            
            Ein Random Forest ist ein Ensemble-Learning-Algorithmus, der aus mehreren Decision Trees besteht. Der Hauptvorteil ist, dass er genauere Ergebnisse liefert als einzelne Decision Trees, besonders wenn die Bäume nicht miteinander korrelieren.
            
            ## Funktionsprinzip
            
            ```python
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
            ```
            
            ### Wichtige Parameter
            
            - n_estimators: Anzahl der Bäume im Wald
            - max_depth: Maximale Tiefe der einzelnen Bäume
            - min_samples_split: Minimale Anzahl von Samples für eine Aufteilung
            - bootstrap: Ob Bootstrapping verwendet werden soll
            
            ### Vorteile des Random Forest
            
            - Reduziert Overfitting durch Ensemble-Ansatz
            - Kann Feature Importance bestimmen
            - Robust gegen Ausreißer und Rauschen
            - Parallelisierbar und damit effizient bei großen Datensätzen
        - Neural Net
            
            Ein neuronales Netz ist ein mathematisches Modell, das vom menschlichen Gehirn inspiriert wurde. Es besteht aus miteinander verbundenen Neuronen, die in Schichten angeordnet sind und Informationen verarbeiten.
            
            ## Grundlegende Struktur
            
            - Eingabeschicht (Input Layer): Nimmt die Rohdaten auf
            - Versteckte Schichten (Hidden Layers): Verarbeiten die Informationen
            - Ausgabeschicht (Output Layer): Liefert das Endergebnis
            
            ## Mathematische Grundlagen
            
            Jedes Neuron führt folgende Operationen durch:
            
            $$
            output = f(Σ(wi * xi + b))
            $$
            
            wobei:
            
            - wi = Gewichte
            - xi = Eingabewerte
            - b = Bias
            - f = Aktivierungsfunktion
            
            ## Trainingsalgorithmus (Backpropagation)
            
            1. Forward Pass: Berechnung der Ausgabe für gegebene Eingaben
            2. Fehlerberechnung: Vergleich zwischen Vorhersage und tatsächlichem Wert
            3. Backward Pass: Berechnung der Gradienten durch Kettenregel
            4. Gewichtsanpassung: Update der Gewichte mittels Gradientenabstieg
            
            ## Implementierungsbeispiel in Python
            
            ```python
            import tensorflow as tf
            from tensorflow.keras import layers, models
            import numpy as np
            
            # Einfaches neuronales Netz erstellen
            model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=(10,)),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            # Modell kompilieren
            model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
            
            # Beispieldaten generieren
            X_train = np.random.random((1000, 10))
            y_train = np.random.randint(2, size=(1000, 1))
            
            # Modell trainieren
            history = model.fit(X_train, y_train, 
                               epochs=10, 
                               batch_size=32,
                               validation_split=0.2)
            
            # Vorhersagen machen
            predictions = model.predict(X_train[:5])
            print("Vorhersagen:", predictions)
            ```
            
            Hier ist ein Beispiel für ein neuronales Netz mit scikit-learn:
            
            ```python
            from sklearn.neural_network import MLPClassifier
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            
            # Beispieldaten generieren
            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
            
            # Daten aufteilen
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Neuronales Netz erstellen
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32),  # zwei versteckte Schichten
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
            
            # Model trainieren
            mlp.fit(X_train, y_train)
            
            # Genauigkeit ausgeben
            print(f"Genauigkeit auf Testdaten: {mlp.score(X_test, y_test):.3f}")
            
            # Vorhersagen machen
            predictions = mlp.predict(X_test[:5])
            print("Vorhersagen:", predictions)
            ```
            
            Dieses Beispiel verwendet MLPClassifier aus scikit-learn und folgt einer ähnlichen Struktur wie das TensorFlow-Beispiel, ist aber etwas einfacher in der Handhabung.
            
            ## Wichtige Hyperparameter
            
            - Lernrate: Bestimmt die Schrittgröße bei der Gewichtsanpassung
            - Batch-Größe: Anzahl der Trainingsbeispiele pro Gradient-Update
            - Epochen: Anzahl der kompletten Durchläufe durch den Datensatz
            - Netzwerkarchitektur: Anzahl und Größe der versteckten Schichten
            
            ## Vor- und Nachteile
            
            Vorteile:
            
            - Hohe Flexibilität und Ausdrucksstärke
            - Gut geeignet für komplexe Muster
            - Automatische Feature-Extraktion
            
            Nachteile:
            
            - Benötigt große Datenmengen
            - Rechenintensives Training
            - Schwierige Interpretierbarkeit ("Black Box")
        - AdaBoost
            
            AdaBoost (Adaptive Boosting) ist ein Ensemble-Learning-Algorithmus, der schwache Lerner zu einem starken Klassifikator kombiniert. Der Algorithmus passt die Gewichtung der Trainingsdaten nach jeder Iteration an, wobei falsch klassifizierte Beispiele ein höheres Gewicht erhalten.
            
            ## Mathematische Grundlagen
            
            Für jeden schwachen Lerner ht wird ein Gewicht αt berechnet:
            
            ```python
            αt = 0.5 * ln((1 - εt) / εt)
            
            ```
            
            wobei εt der gewichtete Fehler des Klassifikators ist.
            
            Die finale Klassifikation erfolgt durch:
            
            ```python
            H(x) = sign(Σ αt * ht(x))
            
            ```
            
            ## Implementierungsbeispiel
            
            ```python
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
            
            ```
            
            ## Wichtige Parameter
            
            - base_estimator: Der schwache Lerner (meist Decision Trees)
            - n_estimators: Anzahl der schwachen Lerner im Ensemble
            - learning_rate: Kontrolliert den Beitrag jedes Klassifikators
            
            ## Vor- und Nachteile
            
            Vorteile:
            
            - Einfach zu implementieren
            - Gute Generalisierung
            - Weniger anfällig für Overfitting als einzelne Decision Trees
            
            Nachteile:
            
            - Sensibel gegenüber Rauschen und Ausreißern
            - Sequentieller Algorithmus (schwer zu parallelisieren)
            - Kann bei schwachen Basislernern rechenintensiv sein
        - Naive Bayes
            
            Der Naive Bayes-Klassifikator ist ein überwachter Algorithmus für maschinelles Lernen, der für Klassifizierungsaufgaben wie Textklassifizierung verwendet wird. Er nutzt die Prinzipien der Wahrscheinlichkeit, um Klassifizierungsaufgaben durchzuführen.
            
            Gehört zu der Familie der generativen Lernalgorithmen. Er versucht also die Verteilung der Eingaben einer bestimmten Klasse oder Kategorie zu modellieren. Im Gegensatz zu diskriminierenden Klassifikatoren wie der logistischen Regression lernt er nicht, welche Merkmale für die Unterscheidung zwischen den Klassen am wichtigsten sind.
            
            Mathematische Grundlagen:
            
            Der Naive Bayes-Klassifikator basiert auf dem Bayes'schen Theorem:
            
            ```python
            P(y|X) = P(X|y) * P(y) / P(X)
            ```
            
            Dabei ist:
            
            - P(y|X): Die Wahrscheinlichkeit der Klasse y gegeben die Features X
            - P(X|y): Die Wahrscheinlichkeit der Features gegeben die Klasse
            - P(y): Die Prior-Wahrscheinlichkeit der Klasse
            - P(X): Die Evidenz
            
            Hier ein Implementierungsbeispiel:
            
            ```python
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
            
            ```
            
    - Diskriminanzanalyse
        
        Diskriminanzanalysen sind eine Methode der multivarianten Verfahren in der Statistik und dient der Unterscheidung von zwei oder mehr Gruppen, die mit mehreren Merkmalen auch Variablen beschrieben werden. Die Diskriminanzanalyse ist besonders nützlich, wenn es darum geht, die Unterschiede zwischen vordefinierten Gruppen zu maximieren und gleichzeitig die Unterschiede innerhalb der Gruppen zu minimieren. Dabei werden Diskriminanzfunktionen erstellt, die eine optimale Trennung der Gruppen ermöglichen.
        
        - **Linear Diskriminanzanalyse** (LDA)
            
            Ist ein Ansatz, der beim überwachten maschinellen Lernen mehrere Klassen verwendet. Die LDA trennt mehrere Klassen mit mehreren Merkmalen durch Reduzierung der Daten-Dimensionalität.
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%205.png)
            
            LDA arbeitet mit der Identifizierung einer linearen Kombination von Funktionen, die zwei oder mehr Klassen von Objekten oder Ereignissen voneinander trennt oder charakterisiert. Sie projiziert Daten mit zwei oder mehr Dimensionen in eine Dimension, damit sie einfach klassifiziert werden können. Wird auch Reduzierung der Dimensionalität bezeichnet. Die LDA kann für Datenklassifizierungen mit mehreren Klassen verwendet werden um die Funktionszweise anderer Klassifikationsalgorithmen beim lernen zu verbessern.
            
            Die LDA basiert auf der Annahme, dass die Daten normalverteilt sind und alle Klassen die gleiche Kovarianzmatrix haben. Der Algorithmus berechnet die optimale Trennebene zwischen den Klassen durch Maximierung des Verhältnisses der Zwischen-Klassen-Streuung zur Inner-Klassen-Streuung. Ein wichtiger Vorteil der LDA ist ihre Robustheit gegenüber leicht verrauschten Daten.
            
            ```mermaid
            graph TD
                A["Start: Dateneingabe"] --> B["Datenvorverarbeitung"]
                B --> C["Berechnung der Mittelwerte für jede Klasse"]
                C --> D["Berechnung der Kovarianzmatrix"]
                D --> E["Berechnung der Zwischen-Klassen-Streuung"]
                D --> F["Berechnung der Inner-Klassen-Streuung"]
                E --> G["Optimale Projektionsmatrix"]
                F --> G
                G --> H["Dimensionsreduktion"]
                H --> I["Klassifikation"]
                I --> J["Ende: Vorhersage"]
            
                %% Hauptschritte des LDA-Algorithmus
                %% Zwischen-Klassen-Streuung maximieren
                %% Inner-Klassen-Streuung minimieren
            ```
            
            Der Ablauf des LDA-Algorithmus zeigt die schrittweise Verarbeitung von der Dateneingabe bis zur finalen Klassifikation. Besonders wichtig ist die parallele Berechnung der Zwischen-Klassen-Streuung und Inner-Klassen-Streuung, die zusammen zur optimalen Projektionsmatrix führen.
            
            ```jsx
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
            
            ```
            
            Beispielhafter Code in Python der eine LDA an einem Datensatz durchführt.
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%206.png)
            
        - **Quadratische Diskriminanzanalyse** (QDA)
            
            Ähnlich wie bei der **Linearen** Diskriminanzanalyse wird dieses Verfahren verwendet Objekte oder Ereignisse in vordefinierte Klassen einzuordenen. Es basiert wieder auf der Annahme das die Daten Normalverteilt sind und unterscheidet sich darin das die Grenzen zwischen den Klassen auch nicht-linear sein können.
            
            Die QDA geht davon aus, dass jede Klasse ihre eigene Kovarianzmatrix hat, wodurch sie flexibler als die LDA ist. Dies ermöglicht eine bessere Anpassung an komplexere Datenstrukturen, führt aber auch zu einem höheren Risiko des Overfittings. Die Entscheidungsgrenzen der QDA sind quadratisch (gekrümmt), was besonders bei nicht-linearen Klassentrennungen vorteilhaft ist.
            
            Beispiel Code:
            
            ```jsx
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
            ```
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%207.png)
            
            ```mermaid
            graph TD
                A["Start: Dateneingabe"] --> B["Datenvorverarbeitung"]
                B --> C["Berechnung der klassenspezifischen Mittelwerte"]
                C --> D["Berechnung separater Kovarianzmatrizen für jede Klasse"]
                D --> E["Schätzung der Prior-Wahrscheinlichkeiten"]
                E --> F["Quadratische Diskriminanzfunktion berechnen"]
                F --> G["Klassifikation neuer Datenpunkte"]
                G --> H["Berechnung der posteriori Wahrscheinlichkeiten"]
                H --> I["Klassenzuweisung basierend auf max. Wahrscheinlichkeit"]
                I --> J["Ende: Vorhersage"]
            
                %% QDA-Algorithmus Hauptschritte
                %% Unterschied zur LDA: Separate Kovarianzmatrizen
                %% Ermöglicht nicht-lineare Entscheidungsgrenzen
            ```
            
            Die mathematische Grundlage der QDA basiert auf dem Bayes-Theorem und der Annahme multivariater Normalverteilungen für jede Klasse k:
            
            ```python
            P(y=k|x) ∝ P(x|y=k)P(y=k)
            ```
            
            Dabei wird die Likelihood P(x|y=k) für jede Klasse k als multivariate Normalverteilung modelliert:
            
            ```python
            P(x|y=k) = (2π)^(-p/2)|Σₖ|^(-1/2)exp(-1/2(x-μₖ)ᵀΣₖ⁻¹(x-μₖ))
            ```
            
            Wobei:
            
            - μₖ = Mittelwertvektor der Klasse k
            - Σₖ = Kovarianzmatrix der Klasse k
            - p = Anzahl der Merkmale
            - |Σₖ| = Determinante der Kovarianzmatrix
            
            Die quadratische Diskriminanzfunktion δₖ(x) für jede Klasse k ergibt sich durch Logarithmieren:
            
            ```python
            δₖ(x) = -1/2log|Σₖ| - 1/2(x-μₖ)ᵀΣₖ⁻¹(x-μₖ) + log(πₖ)
            ```
            
            Dabei ist πₖ die Prior-Wahrscheinlichkeit der Klasse k. Ein neuer Datenpunkt x wird der Klasse mit dem größten Wert δₖ(x) zugeordnet.
            
            Der Hauptunterschied zur LDA liegt darin, dass bei der QDA für jede Klasse k eine eigene Kovarianzmatrix Σₖ geschätzt wird, während die LDA eine gemeinsame Kovarianzmatrix für alle Klassen verwendet. Dies führt zu den charakteristischen quadratischen (nicht-linearen) Entscheidungsgrenzen der QDA.
            
        
        ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%208.png)
        
        Zunächst ist festzuhalten, dass LDA und QDA für den ersten und zweiten Datensatz äquivalent sind. Der Hauptunterschied besteht darin, dass die LDA annimmt, dass die Kovarianzmatrix jeder Klasse gleich ist, während die QDA eine Kovarianzmatrix pro Klasse schätzt. Da in diesen Fällen der datengenerierende Prozess für beide Klassen die gleiche Kovarianzmatrix hat, schätzt die QDA zwei Kovarianzmatrizen, die (fast) gleich sind und daher der von der LDA geschätzten Kovarianzmatrix entsprechen.
        
        Im ersten Datensatz ist die Kovarianzmatrix, die zur Erstellung des Datensatzes verwendet wurde, sphärisch, was zu einer Diskriminanzgrenze führt, die mit der senkrechten Winkelhalbierenden zwischen den beiden Mittelwerten übereinstimmt. Dies ist beim zweiten Datensatz nicht mehr der Fall. Die Diskriminanzgrenze geht nur durch die Mitte der beiden Mittelwerte.
        
        Im dritten Datensatz schließlich sehen wir den eigentlichen Unterschied zwischen LDA und QDA. Die QDA passt sich an zwei Kovarianzmatrizen an und liefert eine nichtlineare Diskriminanzgrenze, während die LDA unterdurchschnittlich abschneidet, da sie davon ausgeht, dass die beiden Klassen eine einzige Kovarianzmatrix teilen.
        
        - Code dazu
            
            ```jsx
            """
            ====================================================================
            Linear and Quadratic Discriminant Analysis with covariance ellipsoid
            ====================================================================
            
            This example plots the covariance ellipsoids of each class and the decision boundary
            learned by :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis` (LDA) and
            :class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` (QDA). The
            ellipsoids display the double standard deviation for each class. With LDA, the standard
            deviation is the same for all the classes, while each class has its own standard
            deviation with QDA.
            """
            
            # Authors: The scikit-learn developers
            # SPDX-License-Identifier: BSD-3-Clause
            
            # %%
            # Data generation
            # ---------------
            #
            # First, we define a function to generate synthetic data. It creates two blobs centered
            # at `(0, 0)` and `(1, 1)`. Each blob is assigned a specific class. The dispersion of
            # the blob is controlled by the parameters `cov_class_1` and `cov_class_2`, that are the
            # covariance matrices used when generating the samples from the Gaussian distributions.
            import numpy as np
            
            def make_data(n_samples, n_features, cov_class_1, cov_class_2, seed=0):
                rng = np.random.RandomState(seed)
                X = np.concatenate(
                    [
                        rng.randn(n_samples, n_features) @ cov_class_1,
                        rng.randn(n_samples, n_features) @ cov_class_2 + np.array([1, 1]),
                    ]
                )
                y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
                return X, y
            
            # %%
            # We generate three datasets. In the first dataset, the two classes share the same
            # covariance matrix, and this covariance matrix has the specificity of being spherical
            # (isotropic). The second dataset is similar to the first one but does not enforce the
            # covariance to be spherical. Finally, the third dataset has a non-spherical covariance
            # matrix for each class.
            covariance = np.array([[1, 0], [0, 1]])
            X_isotropic_covariance, y_isotropic_covariance = make_data(
                n_samples=1_000,
                n_features=2,
                cov_class_1=covariance,
                cov_class_2=covariance,
                seed=0,
            )
            covariance = np.array([[0.0, -0.23], [0.83, 0.23]])
            X_shared_covariance, y_shared_covariance = make_data(
                n_samples=300,
                n_features=2,
                cov_class_1=covariance,
                cov_class_2=covariance,
                seed=0,
            )
            cov_class_1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0
            cov_class_2 = cov_class_1.T
            X_different_covariance, y_different_covariance = make_data(
                n_samples=300,
                n_features=2,
                cov_class_1=cov_class_1,
                cov_class_2=cov_class_2,
                seed=0,
            )
            
            # %%
            # Plotting Functions
            # ------------------
            #
            # The code below is used to plot several pieces of information from the estimators used,
            # i.e., :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis` (LDA) and
            # :class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` (QDA). The
            # displayed information includes:
            #
            # - the decision boundary based on the probability estimate of the estimator;
            # - a scatter plot with circles representing the well-classified samples;
            # - a scatter plot with crosses representing the misclassified samples;
            # - the mean of each class, estimated by the estimator, marked with a star;
            # - the estimated covariance represented by an ellipse at 2 standard deviations from the
            #   mean.
            import matplotlib as mpl
            from matplotlib import colors
            
            from sklearn.inspection import DecisionBoundaryDisplay
            
            def plot_ellipse(mean, cov, color, ax):
                v, w = np.linalg.eigh(cov)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                # filled Gaussian at 2 standard deviation
                ell = mpl.patches.Ellipse(
                    mean,
                    2 * v[0] ** 0.5,
                    2 * v[1] ** 0.5,
                    angle=180 + angle,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=2,
                )
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.4)
                ax.add_artist(ell)
            
            def plot_result(estimator, X, y, ax):
                cmap = colors.ListedColormap(["tab:red", "tab:blue"])
                DecisionBoundaryDisplay.from_estimator(
                    estimator,
                    X,
                    response_method="predict_proba",
                    plot_method="pcolormesh",
                    ax=ax,
                    cmap="RdBu",
                    alpha=0.3,
                )
                DecisionBoundaryDisplay.from_estimator(
                    estimator,
                    X,
                    response_method="predict_proba",
                    plot_method="contour",
                    ax=ax,
                    alpha=1.0,
                    levels=[0.5],
                )
                y_pred = estimator.predict(X)
                X_right, y_right = X[y == y_pred], y[y == y_pred]
                X_wrong, y_wrong = X[y != y_pred], y[y != y_pred]
                ax.scatter(X_right[:, 0], X_right[:, 1], c=y_right, s=20, cmap=cmap, alpha=0.5)
                ax.scatter(
                    X_wrong[:, 0],
                    X_wrong[:, 1],
                    c=y_wrong,
                    s=30,
                    cmap=cmap,
                    alpha=0.9,
                    marker="x",
                )
                ax.scatter(
                    estimator.means_[:, 0],
                    estimator.means_[:, 1],
                    c="yellow",
                    s=200,
                    marker="*",
                    edgecolor="black",
                )
            
                if isinstance(estimator, LinearDiscriminantAnalysis):
                    covariance = [estimator.covariance_] * 2
                else:
                    covariance = estimator.covariance_
                plot_ellipse(estimator.means_[0], covariance[0], "tab:red", ax)
                plot_ellipse(estimator.means_[1], covariance[1], "tab:blue", ax)
            
                ax.set_box_aspect(1)
                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set(xticks=[], yticks=[])
            
            # %%
            # Comparison of LDA and QDA
            # -------------------------
            #
            # We compare the two estimators LDA and QDA on all three datasets.
            import matplotlib.pyplot as plt
            
            from sklearn.discriminant_analysis import (
                LinearDiscriminantAnalysis,
                QuadraticDiscriminantAnalysis,
            )
            
            fig, axs = plt.subplots(nrows=3, ncols=2, sharex="row", sharey="row", figsize=(8, 12))
            
            lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
            qda = QuadraticDiscriminantAnalysis(store_covariance=True)
            
            for ax_row, X, y in zip(
                axs,
                (X_isotropic_covariance, X_shared_covariance, X_different_covariance),
                (y_isotropic_covariance, y_shared_covariance, y_different_covariance),
            ):
                lda.fit(X, y)
                plot_result(lda, X, y, ax_row[0])
                qda.fit(X, y)
                plot_result(qda, X, y, ax_row[1])
            
            axs[0, 0].set_title("Linear Discriminant Analysis")
            axs[0, 0].set_ylabel("Data with fixed and spherical covariance")
            axs[1, 0].set_ylabel("Data with fixed covariance")
            axs[0, 1].set_title("Quadratic Discriminant Analysis")
            axs[2, 0].set_ylabel("Data with varying covariances")
            fig.suptitle(
                "Linear Discriminant Analysis vs Quadratic Discriminant Analysis",
                y=0.94,
                fontsize=15,
            )
            plt.show()
            
            # %%
            # The first important thing to notice is that LDA and QDA are equivalent for the
            # first and second datasets. Indeed, the major difference is that LDA assumes
            # that the covariance matrix of each class is equal, while QDA estimates a
            # covariance matrix per class. Since in these cases the data generative process
            # has the same covariance matrix for both classes, QDA estimates two covariance
            # matrices that are (almost) equal and therefore equivalent to the covariance
            # matrix estimated by LDA.
            #
            # In the first dataset the covariance matrix used to generate the dataset is
            # spherical, which results in a discriminant boundary that aligns with the
            # perpendicular bisector between the two means. This is no longer the case for
            # the second dataset. The discriminant boundary only passes through the middle
            # of the two means.
            #
            # Finally, in the third dataset, we observe the real difference between LDA and
            # QDA. QDA fits two covariance matrices and provides a non-linear discriminant
            # boundary, whereas LDA underfits since it assumes that both classes share a
            # single covariance matrix.
            
            ```
            
    - Gauß-Prozess
        
        Ist eine nichtparametrische überwachte Lernstrategie, die für Regressions- und probabilistische Klassifikationsprobleme genutzt wird. Die Vorteile von Gauß-Prozessen sind:
        
        | Vorteile | Nachteile |
        | --- | --- |
        | Die Vorhersage interpoliert die Beobachtungen (zumindest bei regulären Kernels) | Implementierung ist nicht sparse (verwendet gesamte Stichproben-/Merkmalsinformationen) |
        | Probabilistische Vorhersage (Gauß-verteilt) mit empirischen Konfidenzintervallen für adaptives Fitting | Verliert Effizienz in hochdimensionalen Räumen (>12 Merkmale) |
        | Vielseitig durch verschiedene verfügbare und benutzerdefinierte Kernels |  |
        
        Der GaußProcessRegressor implementiert Gaußsche Prozesse (GP) für Regressionszwecke. Hierfür muss der Prior des GP angegeben werden. GP kombiniert diesen Prior und die Likelihood-Funktion auf der Grundlage von Trainingsstichproben. Es ermöglicht einen probalistischen Ansatz für die Vorhesrsage, indem er bei der Vorhersage den Mittelwert und die Standardabweichung als Ausgabe liefert.
        
        ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%209.png)
        
        Der Gauß-Prozess verwendet einen flexiblen Ansatz zur Modellierung: Der Prior (Vorwissen) wird entweder als Null (bei normalize_y=False) oder als Durchschnitt der Trainingsdaten (bei normalize_y=True) festgelegt. Die Struktur der Daten wird durch einen Kernel erfasst, der die Kovarianzbeziehungen zwischen den Datenpunkten beschreibt.
        
        Ein besonders wichtiger Aspekt ist die Optimierung der Hyperparameter: Der GaussianProcessRegressor maximiert die Log-Marginal-Likelihood (LML), um die bestmöglichen Kernelparameter zu finden. Da diese Optimierung komplex sein kann und mehrere optimale Lösungen existieren könnten, bietet der Algorithmus die Möglichkeit, die Suche mehrfach mit verschiedenen Startpunkten durchzuführen (gesteuert durch n_restarts_optimizer).
        
        Der Optimierungsprozess läuft dabei in zwei Phasen ab:
        
        - Erste Phase: Verwendung der initial festgelegten Hyperparameter
        - Weitere Phasen: Zufällige Auswahl neuer Startpunkte aus dem erlaubten Wertebereich
        
        Für Anwendungsfälle, in denen die ursprünglichen Hyperparameter beibehalten werden sollen, kann die Optimierung durch Setzen des Optimierers auf None deaktiviert werden. Dies ist besonders nützlich, wenn bereits optimale Parameter bekannt sind oder eine spezifische Konfiguration getestet werden soll.
        
        ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%2010.png)
        
        Der GaussianProcessClassifier (GPC) implementiert Gauß-Prozesse für Klassifizierungsaufgaben, insbesondere für probabilistische Klassifikation. Die wichtigsten Merkmale sind:
        
        - Erzeugt Klassenwahrscheinlichkeiten als Vorhersagen
        - Verwendet eine latente Funktion mit GP-Prior, die durch eine Link-Funktion in Wahrscheinlichkeiten umgewandelt wird
        - Implementiert die logistische Link-Funktion für binäre Klassifikation
        
        Besonderheiten der Implementation:
        
        - Die posteriore Verteilung ist nicht-Gaussisch aufgrund der diskreten Klassenlabels
        - Verwendet die Laplace-Approximation zur Annäherung der posterioren Verteilung
        - Unterstützt Mehrklassen-Klassifikation durch one-vs-rest oder one-vs-one Strategien
        
        Wichtige praktische Aspekte:
        
        - Der Prior-Mittelwert wird als Null angenommen
        - Die Hyperparameter werden durch Maximierung der Log-Marginal-Likelihood optimiert
        - One-vs-one kann bei großen Datensätzen rechnerisch effizienter sein, unterstützt aber keine Wahrscheinlichkeitsschätzungen
        
        Die Implementierung basiert auf der Lösung mehrerer binärer Klassifikationsaufgaben, die dann kombiniert werden, da eine echte Mehrklassen-Laplace-Approximation noch nicht implementiert ist.
        
        - **Beispiel**
            
            ### Hintergrund: Was ist ein Gaußprozess?
            
            Ein Gaußprozess ist ein stochastischer Prozess, der eine Verallgemeinerung der gaußschen Verteilungsannahme auf Funktionen darstellt. Statt Punktwerte (wie in einer Gaußverteilung) zu modellieren, modelliert ein Gaußprozess ganze Funktionen. Das heißt, wir nehmen an, dass für jeden Punkt im Eingaberaum (z. B. für jeden Datenpunkt mit bestimmten Merkmalen) die Ausgabe des Modells (z. B. ein vorhergesagter Hauspreis) als Realisierung einer multivariaten Gaußverteilung entspringt. Damit lässt sich nicht nur eine Punktschätzung für die Vorhersage machen, sondern auch eine Unsicherheit in Form von Varianzen angeben.
            
            In der Praxis wird ein Gaußprozess durch einen **Kernel** (auch Kovarianzfunktion genannt) charakterisiert. Der Kernel bestimmt, wie ähnlich zwei Eingabepunkte in Bezug auf ihre Ausgaben betrachtet werden. Beliebte Kernel sind der Radial-Basis-Funktions-Kernel (RBF), Matern-Kernel oder Dot-Product-Kernel.
            
            Der GaussianProcessRegressor in scikit-learn:
            
            - **Training:** Ermittelt die Parameter des Kernels (falls sie optimierbar sind) und passt den Gaußprozess an die Trainingsdaten an.
            - **Vorhersage:** Gibt für neue Eingabepunkte eine Schätzung des Mittelwertes und der Varianz der entsprechenden Gaußverteilung aus. Damit kann man neben der Schätzung selbst auch einen Unsicherheitsbereich angeben.
            - **California Housing Market**
                
                ### Erklärung des Codes
                
                1. **Datensatz laden:**
                    
                    Mit `fetch_california_housing()` holen wir uns einen realen Datensatz, der Informationen zu Immobilienpreisen in Kalifornien enthält. `X` sind die Eingangsmerkmale (z. B. Durchschnittsalter der Gebäude in der Region, Bevölkerungsdichte, etc.), `y` sind die Zielwerte (durchschnittliche Hauspreise).
                    
                2. **Datenaufteilung:**
                    
                    Mit `train_test_split` teilen wir die Daten in Trainings- und Testsatz, um später die Qualität der Vorhersage evaluieren zu können.
                    
                3. **Vorverarbeitung:**
                    
                    Wir standardisieren die Merkmale mit `StandardScaler`, da Gaußprozesse (und viele andere ML-Methoden) oft von skalierten Daten profitieren. Das bedeutet, wir bringen alle Merkmale auf einen ähnlichen Wertebereich und vermeiden damit, dass Merkmale mit großen Skalen dominieren.
                    
                4. **Kernel definieren:**
                    
                    Wir wählen einen Kernel aus, der festlegt, wie die Punkte miteinander in Beziehung stehen. Hier verwenden wir einen RBF-Kernel, der die Ähnlichkeit zweier Punkte als exponentiell fallende Funktion ihrer Distanz betrachtet. Der `ConstantKernel` skaliert diesen Wert zusätzlich.
                    
                5. **Modell erstellen und anpassen:**
                    
                    Der `GaussianProcessRegressor` erhält den Kernel und weitere Parameter (z. B. `alpha` für die Regularisierung). Durch `gp.fit(X_train_scaled, y_train)` wird der Prozess an die Daten angepasst. Dabei optimiert er intern die Hyperparameter des Kernels (wenn möglich).
                    
                6. **Vorhersage treffen:**
                    
                    Mit `gp.predict(X_test_scaled, return_std=True)` erhalten wir sowohl die geschätzten Mittelwerte `y_pred` als auch die Standardabweichungen `y_std` für die Vorhersagen. Dies ist einer der Hauptvorteile von Gaußprozessen, dass wir eine Unsicherheitsschätzung bekommen.
                    
                7. **Bewertung der Güte:**
                    
                    Wir berechnen den MSE (mittleren quadratischen Fehler) auf den Testdaten, um die Genauigkeit der Vorhersagen zu quantifizieren.
                    
                8. **Visualisierung:**
                    
                    Um einen Eindruck von der Modellgüte zu bekommen, plotten wir wahre und vorhergesagte Hauspreise und visualisieren die Unsicherheit als Bänder um die Vorhersagekurve. Ein engeres Unsicherheitsband bedeutet, dass das Modell bei diesen Punkten eine höhere Zuversicht in die eigene Schätzung hat.
                    
                
                ### Zusammenfassung
                
                In diesem Beispiel haben wir einen Gaußprozess zur Regression verwendet, um Hauspreise vorherzusagen. Das Modell nutzt einen Kernel, um Ähnlichkeiten zwischen Datenpunkten abzubilden, und gibt neben den Vorhersagen auch Unsicherheitsmaße aus. Dieses Beispiel zeigt, wie man in Python mittels `scikit-learn` Gaußprozess-Modelle für reale Daten einsetzen kann.
                
                ```python
                #!/usr/bin/env python3
                # -*- coding: utf-8 -*-
                """
                Created on Thu Dec 12 08:42:16 2024
                
                @author: colinengelmann
                """
                
                import numpy as np
                from sklearn.datasets import fetch_california_housing
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
                import matplotlib.pyplot as plt
                
                # 1. Datensatz laden
                data = fetch_california_housing()
                X = data.data  # Merkmale der Häuser
                y = data.target  # Zielvariable: Durchschnittlicher Hauspreis in der Region
                
                # Für bessere Interpretierbarkeit: Wir nutzen nur ein Teilset der Daten
                # um die Laufzeit zu verkürzen.
                X = X[:2000]
                y = y[:2000]
                
                # 2. Daten aufteilen in Trainings- und Testset
                X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                    test_size=0.2, 
                                                                    random_state=42)
                
                # 3. Vorverarbeitung: Standardisierung
                scaler_X = StandardScaler()
                X_train_scaled = scaler_X.fit_transform(X_train)
                X_test_scaled = scaler_X.transform(X_test)
                
                # 4. Kernel definieren: Hier verwenden wir einen RBF-Kernel mit einer Anfangsvarianz.
                # ConstantKernel(c, ...) liefert einen skalierenden Faktor, RBF(l) definiert den "Längenmaßstab".
                kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                
                # 5. GaussianProcessRegressor erstellen und anpassen
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1, random_state=42)
                gp.fit(X_train_scaled, y_train)
                
                # 6. Vorhersage auf dem Testset
                y_pred, y_std = gp.predict(X_test_scaled, return_std=True)
                
                # 7. Gütebewertung
                mse = np.mean((y_test - y_pred)**2)
                print("Mittlerer quadratischer Fehler (MSE):", mse)
                
                # 8. Beispielhafter Plot:
                # Wir plotten die vorhergesagten Werte gegen die wahren Werte.
                # Zusätzlich zeigen wir die Unsicherheit (Standardabweichung) mit Fehlerschläuchen an.
                
                # Da unser Datensatz mehrdimensional ist, wählen wir z. B. eine Teilmenge von Testpunkten für ein Plot.
                # Wir sortieren nach den wahren Werten, um einen übersichtlicheren Plot zu erhalten.
                sort_idx = np.argsort(y_test)
                y_test_sorted = y_test[sort_idx]
                y_pred_sorted = y_pred[sort_idx]
                y_std_sorted = y_std[sort_idx]
                
                plt.figure(figsize=(10, 6))
                plt.plot(y_test_sorted, label="Wahre Werte", marker='o', linestyle=':', color='blue')
                plt.plot(y_pred_sorted, label="Vorhersage GP", marker='x', linestyle='-', color='red')
                
                # Unsicherheitsbänder um die Vorhersage
                plt.fill_between(range(len(y_pred_sorted)),
                                 y_pred_sorted - 2*y_std_sorted,
                                 y_pred_sorted + 2*y_std_sorted,
                                 alpha=0.2, color='red', label='Unsicherheit (±2 Std)')
                
                plt.xlabel("Testdatenpunkte (sortiert nach wahrem Wert)")
                plt.ylabel("Hauspreis (in 100.000 USD)")
                plt.title("Gaussian Process Regression auf California Housing Daten")
                plt.legend()
                plt.show()
                
                ```
                
                Das verwendete Modell – in diesem Fall ein Gaußprozessregressor – nimmt die Merkmale (Eingangsvariablen) des California-Housing-Datensatzes als Eingabe und gibt eine Schätzung für den mittleren Hauspreis in der entsprechenden Region aus. Konkret bedeutet das:
                
                1. **Eingangsmerkmale:**
                    
                    Der California Housing Datensatz beinhaltet verschiedene Merkmale einer geografischen Region in Kalifornien, z. B.:
                    
                    - **MedInc:** Medianes Einkommen der Haushalte in der Region.
                    - **HouseAge:** Durchschnittliches Baujahr bzw. Alter der Häuser.
                    - **AveRooms:** Durchschnittliche Anzahl an Zimmern pro Haushalt.
                    - **AveBedrms:** Durchschnittliche Anzahl an Schlafzimmern pro Haushalt.
                    - **Population:** Bevölkerungszahl der Region.
                    - **AveOccup:** Durchschnittliche Anzahl an Personen pro Haushalt.
                    - **Latitude und Longitude:** Geographische Koordinaten der Region.
                    
                    Jedes dieser Merkmale für eine bestimmte Region (oder ein bestimmtes Cluster von Haushalten) dient als Eingabevektor für das Modell.
                    
                2. **Ausgabe des Modells (Zielvariable):**
                    
                    Die Zielgröße, die das Modell vorhersagen soll, ist der durchschnittliche Hauspreis in dieser Region. Er wird in hunderttausend US-Dollar (100.000 $) angegeben. Wenn das Modell zum Beispiel einen Wert von 2,5 vorhersagt, entspricht dies einem durchschnittlichen Hauspreis von 250.000 US-Dollar.
                    
                3. **Was macht das Modell genau?**
                    
                    Das Gaußprozessmodell versucht, einen Zusammenhang zwischen den Eingangsmerkmalen (Einkommen, Zimmeranzahl, geografische Lage etc.) und dem durchschnittlichen Hauspreis herzustellen.
                    
                    - Wenn man dem Modell einen neuen Datenpunkt gibt (z. B. eine Region mit bestimmtem durchschnittlichem Einkommen, bestimmter Bevölkerungszahl, bestimmten geographischen Koordinaten, etc.), dann schätzt es anhand der gelernten Zusammenhänge den mittleren Hauswert für diese Region.
                    - Neben einer reinen Punktschätzung (ein Wert für den Preis) liefert der Gaußprozess zudem eine Unsicherheitsabschätzung. Das heißt, er gibt nicht nur die "beste Schätzung" aus, sondern auch ein Maß, wie sicher oder unsicher er sich dabei ist.
                
                Insgesamt nimmt das Modell also einen Merkmalsvektor (basierend auf Geographie, Demographie und Gebäudestruktur) und sagt vorher, wie teuer das Haus voraussichtlich sein wird, basierend auf den Mustern, die es während des Trainings aus echten Daten gelernt hat.
                
            - **GP on Iris Datenset**
                
                Dieser Code demonstriert die Anwendung eines GaussianProcessClassifiers (Gaußprozess-Klassifikators) auf den Iris-Datensatz unter Verwendung zweier verschiedener Kernel-Konfigurationen – einer isotropen und einer anisotropen RBF-Kernelvariante – und visualisiert die resultierenden Klassifikationsgrenzen sowie die Wahrscheinlichkeiten für jede Klasse im zweidimensionalen Merkmalsraum.
                
                **Was passiert Schritt für Schritt?**
                
                1. **Daten laden und Vorverarbeiten:**
                    - Mit `iris = datasets.load_iris()` wird der Iris-Datensatz geladen, der Informationen zu drei Iris-Blumenarten enthält.
                    - `X = iris.data[:, :2]` wählt nur die ersten beiden Merkmale aus, hier sind das meist `Sepal length` und `Sepal width` (Kelchblattlänge und -breite). Dadurch können wir die Daten in einem zweidimensionalen Plot darstellen.
                    - `y = np.array(iris.target, dtype=int)` liefert die zugehörigen Klassenlabels als ganze Zahlen (0, 1, 2) für die drei Iris-Arten.
                2. **Definition der Kernel und Modelltraining:**
                    - Es werden zwei Gaußprozess-Klassifikatoren aufgestellt.
                        - Der erste `gpc_rbf_isotropic` verwendet einen isotropen RBF-Kernel. "Isotrop" bedeutet, dass der Kernel in allen Richtungen im Merkmalsraum gleich "schnell" abfällt. Hier wird der Kernel mit `RBF([1.0])` definiert, was einen einzelnen Längenskalenparameter verwendet.
                        - Der zweite `gpc_rbf_anisotropic` verwendet einen anisotropen RBF-Kernel mit `RBF([1.0, 1.0])`. Das bedeutet, dass der Kernel für jede Richtung im Merkmalsraum (also für jede Eingangsvariable) eine eigene Längenskala hat. So kann der Kernel in einer Merkmalsdimension anders "skaliert" sein als in der anderen.
                    
                    Beide Modelle werden anschließend mit `fit(X, y)` an den Daten trainiert.
                    
                3. **Erzeugen eines Rasters (Mesh) zur Visualisierung:**
                    - Mit `np.meshgrid` wird ein Gitter von Punkten über den Bereich der betrachteten Merkmalswerte erstellt. Dieses Gitter deckt den gesamten Merkmalsraum ab, in dem unsere Trainingspunkte liegen (leicht vergrößert, damit man die Klassifikationsgrenzen schön sieht).
                    - Das Raster dient dazu, die Vorhersage des Modells an vielen Punkten abzurufen, um anschließend die Entscheidungsgrafik (Karten der Klassenwahrscheinlichkeiten) plotten zu können.
                4. **Vorhersage und Wahrscheinlichkeitsdarstellung:**
                    - Für alle Punkte im Mesh wird `clf.predict_proba(...)` aufgerufen, um die Wahrscheinlichkeiten für jede Klasse vorherzusagen.
                    - Die Vorhersagen werden anschließend zu einer Form umgewandelt, mit der `imshow` umgehen kann. So entsteht eine farbige Karte, auf der jeder Punkt im Merkmalsraum eine Farbgebung entsprechend der Klassenwahrscheinlichkeiten erhält.
                5. **Plotten der Ergebnisse:**
                    - Der Code erzeugt zwei nebeneinander liegende Plots (`plt.subplot(1, 2, ...)`), einen für den isotropen und einen für den anisotropen Kernel.
                    - `plt.imshow()` zeigt die Wahrscheinlichkeitsverteilung der Klassen im Hintergrund an.
                    - Mit `plt.scatter()` werden die Trainingspunkte (unsere Iris-Blumen), eingefärbt nach ihrer wahren Klasse (rot, grün, blau), in den Plot eingefügt.
                    - Die Achsen werden entsprechend bezeichnet (`Sepal length`, `Sepal width`), und die Titel enthalten den Namen des Kernels sowie den Wert der "Log Marginal Likelihood" (LML). Die LML ist ein Indikator, wie gut das Modell zu den Daten passt.
                6. **Vergleich der beiden Modelle:**
                    - Durch den Vergleich der beiden Subplots sieht man, wie unterschiedlich isotrope und anisotrope RBF-Kernel den Raum "krümmen" und wie sich dadurch die Entscheidungsgrafiken unterscheiden.
                    - Das anisotrope Modell kann in Richtung eines Merkmals stärker oder schwächer variieren als in Richtung des anderen, was potenziell zu besseren Anpassungen führt, falls die Daten in einer Dimension sensibler auf Veränderungen reagieren als in der anderen.
                
                **Kurz gesagt:**
                
                - Der Code führt eine Gaußprozess-Klassifikation auf den Iris-Daten durch.
                - Er verwendet zwei verschiedene Kernel, um die Auswirkung auf die Klassifikationslandschaft zu zeigen.
                - Das Ergebnis ist eine visuelle Darstellung der Wahrscheinlichkeiten für jede Klasse im Merkmalsraum, zusammen mit den Trainingsdatenpunkten und einem Vergleich der Modellgüte über die log-marginalen Likelihoods.
                - Code
                    
                    ```python
                    #!/usr/bin/env python3
                    # -*- coding: utf-8 -*-
                    """
                    Created on Thu Dec 12 09:02:37 2024
                    
                    @author: colinengelmann
                    """
                    
                    # Authors: The scikit-learn developers
                    # SPDX-License-Identifier: BSD-3-Clause
                    
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    from sklearn import datasets
                    from sklearn.gaussian_process import GaussianProcessClassifier
                    from sklearn.gaussian_process.kernels import RBF
                    
                    # import some data to play with
                    iris = datasets.load_iris()
                    X = iris.data[:, :2]  # we only take the first two features.
                    y = np.array(iris.target, dtype=int)
                    
                    h = 0.02  # step size in the mesh
                    
                    kernel = 1.0 * RBF([1.0])
                    gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)
                    kernel = 1.0 * RBF([1.0, 1.0])
                    gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)
                    
                    # create a mesh to plot in
                    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                    
                    titles = ["Isotropic RBF", "Anisotropic RBF"]
                    plt.figure(figsize=(10, 5))
                    for i, clf in enumerate((gpc_rbf_isotropic, gpc_rbf_anisotropic)):
                        # Plot the predicted probabilities. For that, we will assign a color to
                        # each point in the mesh [x_min, m_max]x[y_min, y_max].
                        plt.subplot(1, 2, i + 1)
                    
                        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
                    
                        # Put the result into a color plot
                        Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
                        plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")
                    
                        # Plot also the training points
                        plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
                        plt.xlabel("Sepal length")
                        plt.ylabel("Sepal width")
                        plt.xlim(xx.min(), xx.max())
                        plt.ylim(yy.min(), yy.max())
                        plt.xticks(())
                        plt.yticks(())
                        plt.title(
                            "%s, LML: %.3f" % (titles[i], clf.log_marginal_likelihood(clf.kernel_.theta))
                        )
                    
                    plt.tight_layout()
                    plt.show()
                    ```
                    
                
                ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%2011.png)
                
- Unsupervised Learning
    - Clustering
        
        Clustering ist ein unüberwachtes Lernverfahren im Machine Learning, bei dem ähnliche Datenpunkte in Gruppen (Cluster) zusammengefasst werden. Das Ziel ist es, Strukturen und Muster in den Daten zu erkennen, ohne dass vorher bekannte Kategorien oder Labels existieren.
        
        Die wichtigsten Eigenschaften von Clustering sind:
        
        - Datenpunkte innerhalb eines Clusters sollten möglichst ähnlich zueinander sein
        - Datenpunkte aus verschiedenen Clustern sollten möglichst unterschiedlich sein
        - Die Ähnlichkeit wird meist über Distanzmaße (z.B. euklidische Distanz) bestimmt
        
        Häufig verwendete Clustering-Algorithmen sind:
        
        - **K-Means:** Teilt Daten in k vorgegebene Cluster ein, indem Zentroide iterativ optimiert werden
        - **DBSCAN:** Findet Cluster basierend auf der Dichte der Datenpunkte, ohne eine fixe Anzahl vorzugeben
        - **Hierarchisches Clustering:** Erstellt eine Hierarchie von Clustern durch schrittweises Zusammenfügen oder Aufteilen
        
        Clustering wird in vielen Bereichen eingesetzt, z.B. für Kundensegmentierung, Bildverarbeitung oder Anomalieerkennung.
        
        - K-Means
            
            Wenn Pukte und eine ganze Zahl k gegeben sind, zielt der Algorithmus darauf ab, die Punkte in k homogene und kompakte Gruppen, sogenannte Cluster zu unterteilen.
             
            
            Der K-Means Algorithmus funktioniert wie folgt:
            
            1. Initialisierung: k zufällige Zentroide werden im Datenraum platziert
            2. Zuordnung: Jeder Datenpunkt wird dem nächstgelegenen Zentroid zugeordnet
            3. Aktualisierung: Die Position jedes Zentroids wird als Durchschnitt aller ihm zugeordneten Punkte neu berechnet
            4. Wiederholung: Schritte 2 und 3 werden wiederholt, bis sich die Zentroide nicht mehr signifikant verschieben
            
            | Vorteile | Nachteile |
            | --- | --- |
            | Einfach zu implementieren und zu verstehen | Anzahl der Cluster k muss vorab festgelegt werden |
            | Effizient bei großen Datensätzen | Sensibel gegenüber Ausreißern |
            | Garantierte Konvergenz | Ergebnis hängt von der initialen Zentroid-Platzierung ab |
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%2012.png)
            
            Die Qualität des Clusterings wird oft mit der Within-Cluster Sum of Squares (WCSS) gemessen - je kleiner dieser Wert, desto kompakter sind die Cluster.
            
            K-means ist ein iterativer schwerpunktbasierter Clustering-Algorithmus, der einen Datensatz basierend auf dem Abstand zwischen den jeweiligen Schwerpunkten in ähnlichen Gruppen unterteilt. Der Schwerpunkt oder das Clusterzentrum ist je nach den Eigenschaften der Daten entweder der Mittelwert oder der Median aller Punkte innerhalb des Clusters.
            
            K-means Clustering ist einfacher, aber empfindlich auf Anfangsbedingungen reagierender Clustering Algorithmus. Es ist also wichtig die Schwerpunktinitialisierung und die Anzahl der Cluster k zu optimieren, um die aussagekräftigsten Cluster zu erhalten. 
            
            1. Alle Datenpunkte innerhalb eines Clusters sollten ähnlic sein
            2. Cluster sollten sich voneinander unterscheiden
            
            Die Eigenschaften werden durch Minimierung des Intracluster-Abstands und Maximierung des Intercluster-Abstand aller Datenpunkte in einem Datensatz erreicht. Je kompakter und isolierter desto besser.
            
            Codebeispiel vom K-means Algorithmus:
            
            ```jsx
            #!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            """
            Created on Wed Dec 11 11:51:05 2024
            
            @author: colinengelmann
            """
            
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.cluster import KMeans
            from sklearn.datasets import make_blobs
            
            # Beispiel-Daten erzeugen
            np.random.seed(42)
            data, labels = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
            
            # K-means Clustering ausführen
            kmeans = KMeans(n_clusters=4, random_state=0)
            kmeans.fit(data)
            predicted_labels = kmeans.predict(data)
            
            # Cluster-Zentren und Labels visualisieren
            plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, s=50, cmap='viridis')
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Zentren')
            plt.title('K-means Clustering')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()
            
            ```
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%2013.png)
            
        - DBSCAN
            
            **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) ist ein Clustering-Algorithmus, der Datenpunkte aufgrund von Dichteeigenschaften in Cluster einteilt, anstatt wie k-Means lediglich nach Abständen zu Gruppenzentren zu suchen. Er ist besonders gut darin, Cluster beliebiger Form zu finden und Ausreißer (Rauschen) klar zu identifizieren. Im Kern beruht DBSCAN auf den folgenden Ideen und Schritten:
            
            **Grundkonzepte:**
            
            1. **Epsilon (ε)**: Ein Radius, der den lokalen Nachbarschaftsbereich um jeden Datenpunkt definiert.
            2. **MinPts**: Eine Mindestanzahl von Punkten, die in der ε-Nachbarschaft vorhanden sein müssen, damit ein Punkt als „Kernpunkt“ gilt.
            3. **Kernpunkt (Core Point)**: Ein Punkt, der mindestens MinPts Punkte (einschließlich sich selbst) in seinem ε-Umkreis hat.
            4. **Randpunkt (Border Point)**: Ein Punkt, der zwar nicht genügend Nachbarn hat, um selbst Kernpunkt zu sein, jedoch in der ε-Nachbarschaft eines Kernpunktes liegt.
            5. **Rauschen (Noise)**: Punkte, die weder Kernpunkte noch Randpunkte eines Clusters sind. Sie gehören zu keinem Cluster, weil sie nicht dicht genug an anderen Punkten liegen.
            
            **Ablauf des Algorithmus:**
            
            1. **Parameterwahl (ε und MinPts):**
                
                Der Anwender wählt zunächst die beiden Hauptparameter:
                
                - **ε (Epsilon)**: Bestimmt die Nachbarschaftsradiusgröße.
                - **MinPts (Minimum Points)**: Bestimmt, ab wann ein Punkt „genug Nachbarn“ hat, um als Kernpunkt zu gelten.
            2. **Identifikation von Kernpunkten:**
                
                Für jeden Datenpunkt wird bestimmt, wie viele Nachbarn er in einem Umkreis von ε hat.
                
                - Wenn die Anzahl dieser Nachbarn (inklusive des Punktes selbst) mindestens MinPts ist, wird dieser Punkt als Kernpunkt markiert.
            3. **Clusterbildung durch „Reachability“ (Erreichbarkeit):**
                
                Sobald ein Kernpunkt identifiziert wurde, wird von diesem Punkt aus ein Cluster aufgebaut, indem alle Punkte, die sich in seinem ε-Nachbarschaftsumkreis befinden, zum selben Cluster hinzugefügt werden.
                
                - Bei diesem Schritt wird rekursiv vorgegangen: Jeder neu hinzugefügte Kernpunkt im erweiterten Nachbarschaftsbereich fügt wiederum seine ε-Nachbarn hinzu, bis keine weiteren Punkte erreichbar sind.
                - Randpunkte, die zwar nicht selbst Kernpunkte sind, aber in der Nähe mindestens eines Kernpunktes liegen, werden dem Cluster ebenfalls zugeordnet.
            4. **Markierung von Rauschen:**
                
                Punkte, die nicht von Kernpunkten aus erreichbar sind und somit nicht in einem Cluster landen, werden als Rauschen klassifiziert. Diese Punkte stehen isoliert und können als Ausreißer betrachtet werden.
                
            
            **Eigenschaften von DBSCAN:**
            
            - **Beliebige Clusterformen:** Da sich DBSCAN nicht auf eine sphärische Clusterform beschränkt (wie es z. B. bei k-Means der Fall ist), kann es Cluster beliebiger Gestalt erkennen.
            - **Automatische Bestimmung der Clusteranzahl:** Anders als bei k-Means muss bei DBSCAN die Anzahl der Cluster nicht vorher angegeben werden. Die Clusteranzahl ergibt sich aus den Daten.
            - **Robust gegenüber Ausreißern:** Ausreißer werden klar als Noise ausgewiesen und beeinflussen nicht die Cluster-Zentren (da es keine expliziten Zentren gibt).
            
            **Herausforderungen:**
            
            - **Parameterwahl:** Die Wahl von ε und MinPts ist oft nicht trivial und kann stark die Ergebnisse beeinflussen. In der Praxis experimentiert man häufig mit verschiedenen Werten oder nutzt Heuristiken (z. B. k-Distance-Plot), um sinnvolle Parameter zu finden.
            - **Unterschiedliche Dichten:** Wenn der Datensatz Bereiche mit deutlich unterschiedlichen Punktedichten aufweist, kann die globale Wahl eines einheitlichen ε-Werts schwierig sein.
            
            **Fazit:**
            
            DBSCAN gruppiert Daten auf Basis lokaler Punktedichte, erkennt automatisch Cluster unterschiedlicher Form und markiert isolierte Punkte als Rauschen. Es ist besonders nützlich, wenn man Cluster von Ausreißern trennen, unregelmäßig geformte Cluster finden und die Anzahl von Clustern nicht im Voraus festlegen möchte.
            
            ```mermaid
            graph TD
                A["Start"] --> B["Parameter initialisieren"]
                B --> C["Epsilon (ε) und MinPts festlegen"]
                C --> D["Wähle unbesuchten Punkt P"]
                D --> E{"Finde Nachbarn im<br/>ε-Radius"}
                E --> F{"Genügend Nachbarn?<br/>(≥ MinPts)"}
                F -- "Ja" --> G["Markiere P als Kernpunkt"]
                G --> H["Erstelle neuen Cluster"]
                H --> I["Füge Nachbarn zum Cluster hinzu"]
                I --> J["Erweitere Cluster<br/>rekursiv"]
                J --> K["Markiere Punkte als besucht"]
                K --> D
                F -- "Nein" --> L{"Im Radius eines<br/>Kernpunkts?"}
                L -- "Ja" --> M["Markiere als Randpunkt"]
                M --> D
                L -- "Nein" --> N["Markiere als Rauschen"]
                N --> D
                D --> O{"Alle Punkte<br/>besucht?"}
                O -- "Nein" --> D
                O -- "Ja" --> P["Ende"]
            %% DBSCAN Algorithmus Ablaufdiagramm
            %% Zeigt die Hauptschritte des dichtebasierten Clusterings
            %% Unterscheidet zwischen Kernpunkten, Randpunkten und Rauschen
            ```
            
            Im folgenden Code Beispiel werden Punkte in einem Raum erzeugt und mittels DBSCAN in Cluster eingeteilt. 
            
            - Code
                
                ```bash
                """
                ===================================
                Demo of DBSCAN clustering algorithm
                ===================================
                
                DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds core
                samples in regions of high density and expands clusters from them. This
                algorithm is good for data which contains clusters of similar density.
                
                See the :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py` example
                for a demo of different clustering algorithms on 2D datasets.
                
                """
                
                # Authors: The scikit-learn developers
                # SPDX-License-Identifier: BSD-3-Clause
                
                # %%
                # Data generation
                # ---------------
                #
                # We use :class:`~sklearn.datasets.make_blobs` to create 3 synthetic clusters.
                
                from sklearn.datasets import make_blobs
                from sklearn.preprocessing import StandardScaler
                
                centers = [[1, 1], [-1, -1], [1, -1]]
                X, labels_true = make_blobs(
                    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
                )
                
                X = StandardScaler().fit_transform(X)
                
                # %%
                # We can visualize the resulting data:
                
                import matplotlib.pyplot as plt
                
                plt.scatter(X[:, 0], X[:, 1])
                plt.show()
                
                # %%
                # Compute DBSCAN
                # --------------
                #
                # One can access the labels assigned by :class:`~sklearn.cluster.DBSCAN` using
                # the `labels_` attribute. Noisy samples are given the label math:`-1`.
                
                import numpy as np
                
                from sklearn import metrics
                from sklearn.cluster import DBSCAN
                
                db = DBSCAN(eps=0.3, min_samples=10).fit(X)
                labels = db.labels_
                
                # Number of clusters in labels, ignoring noise if present.
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise_ = list(labels).count(-1)
                
                print("Estimated number of clusters: %d" % n_clusters_)
                print("Estimated number of noise points: %d" % n_noise_)
                
                # %%
                # Clustering algorithms are fundamentally unsupervised learning methods.
                # However, since :class:`~sklearn.datasets.make_blobs` gives access to the true
                # labels of the synthetic clusters, it is possible to use evaluation metrics
                # that leverage this "supervised" ground truth information to quantify the
                # quality of the resulting clusters. Examples of such metrics are the
                # homogeneity, completeness, V-measure, Rand-Index, Adjusted Rand-Index and
                # Adjusted Mutual Information (AMI).
                #
                # If the ground truth labels are not known, evaluation can only be performed
                # using the model results itself. In that case, the Silhouette Coefficient comes
                # in handy.
                #
                # For more information, see the
                # :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`
                # example or the :ref:`clustering_evaluation` module.
                
                print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
                print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
                print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
                print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
                print(
                    "Adjusted Mutual Information:"
                    f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
                )
                print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")
                
                # %%
                # Plot results
                # ------------
                #
                # Core samples (large dots) and non-core samples (small dots) are color-coded
                # according to the assigned cluster. Samples tagged as noise are represented in
                # black.
                
                unique_labels = set(labels)
                core_samples_mask = np.zeros_like(labels, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                
                colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        # Black used for noise.
                        col = [0, 0, 0, 1]
                
                    class_member_mask = labels == k
                
                    xy = X[class_member_mask & core_samples_mask]
                    plt.plot(
                        xy[:, 0],
                        xy[:, 1],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor="k",
                        markersize=14,
                    )
                
                    xy = X[class_member_mask & ~core_samples_mask]
                    plt.plot(
                        xy[:, 0],
                        xy[:, 1],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor="k",
                        markersize=6,
                    )
                
                plt.title(f"Estimated number of clusters: {n_clusters_}")
                plt.show()
                
                ```
                
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%2014.png)
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%2015.png)
            
        - MeanShift
            
            Mean Shift ist ein dichtebasierter Clustering-Algorithmus, der Datenpunkte in natürliche Gruppen einteilt, ohne dass die Anzahl der Cluster vorher festgelegt werden muss. Der Algorithmus funktioniert wie folgt:
            
            1. **1. Initialisierung:** Jeder Datenpunkt wird als mögliches Clusterzentrum betrachtet
            2. **2. Kernel-Dichte-Schätzung:** Ein Fenster mit festgelegtem Radius (Bandbreite) wird um jeden Punkt gelegt
            3. **3. Verschiebung:** Punkte werden iterativ zum Schwerpunkt ihrer lokalen Nachbarschaft verschoben
            4. **4. Konvergenz:** Der Prozess wird wiederholt, bis die Punkte sich in Regionen hoher Dichte stabilisieren
            
            Die Besonderheit des Algorithmus liegt in seiner Fähigkeit, Cluster beliebiger Form zu erkennen und robust gegenüber Ausreißern zu sein. Die Wahl der Bandbreite ist dabei der wichtigste Parameter, der die Granularität der gefundenen Cluster bestimmt.
            
            ```mermaid
            graph TD
                A["Start"] --> B["Initialisierung: Jeder Datenpunkt als mögliches Zentrum"]
                B --> C["Fenster mit definierter Bandbreite um jeden Punkt legen"]
                C --> D["Berechne gewichteten Mittelwert der Punkte im Fenster"]
                D --> E["Verschiebe Zentrum zum berechneten Mittelwert"]
                E --> F{"Konvergiert?"}
                F -- "Nein" --> D
                F -- "Ja" --> G["Fasse nahe Zentren zu Clustern zusammen"]
                G --> H["Ende: Finale Cluster identifiziert"]
            %% Meanshift ist ein dichtebasierter Clustering-Algorithmus
            %% Die Bandbreite ist der wichtigste Parameter
            %% Der Prozess konvergiert zu Regionen hoher Dichte
            ```
            
            - Code
                
                ```python
                import numpy as np
                import matplotlib.pyplot as plt
                from sklearn.cluster import MeanShift, estimate_bandwidth
                from sklearn.datasets import make_blobs
                
                # 1. Daten erstellen
                centers = [[2, 2], [8, 8], [5, 5]]
                X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=1.0, random_state=42)
                
                # 2. Bandbreite schätzen
                bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
                
                # 3. MeanShift erstellen und anpassen
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(X)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                
                # 4. Visualisierung der Cluster
                plt.figure(figsize=(8, 6))
                plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
                plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Endgültige Zentren')
                plt.title("MeanShift Clustering")
                plt.legend()
                plt.grid()
                plt.show()
                
                ```
                
            
            ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%2016.png)
            
- Neural Networks
    
    # Einführung in Neuronale Netzwerke
    
    ## Was sind Neuronale Netzwerke?
    
    Neuronale Netzwerke sind eine Methode des maschinellen Lernens, die von der Funktionsweise des menschlichen Gehirns inspiriert ist. Sie bestehen aus einer Struktur von **Neuronen**, die in Schichten organisiert sind, und sind besonders effektiv bei der Analyse und Mustererkennung in großen Datenmengen.
    
    ---
    
    ## Aufbau eines Neuronalen Netzwerks
    
    Ein neuronales Netzwerk besteht aus:
    
    1. **Eingabeschicht (Input Layer):**
        - Nimmt die Eingangsdaten (z. B. Bilder, Text, Zahlen) auf.
    2. **Verborgene Schichten (Hidden Layers):**
        - Diese Schichten führen Berechnungen durch, um Merkmale und Zusammenhänge in den Daten zu erkennen.
        - Bestehen aus vielen Neuronen, die durch **Gewichte** und **Aktivierungsfunktionen** miteinander verbunden sind.
    3. **Ausgabeschicht (Output Layer):**
        - Gibt die Ergebnisse zurück (z. B. Klassifikation, Vorhersagen).
    
    ---
    
    ## Funktionsweise
    
    1. **Datenfluss:**
        - Daten werden von der Eingabeschicht durch die verborgenen Schichten zur Ausgabeschicht weitergeleitet.
        - In jeder Schicht werden Berechnungen durchgeführt:
            
            ```
            z = \\sum (Gewicht \\times Eingang) + Bias
            
            ```
            
    2. **Aktivierungsfunktionen:**
        - Entscheiden, ob ein Neuron "aktiviert" wird (z. B. Sigmoid, ReLU, Softmax).
    3. **Training:**
        - Während des Trainings lernt das Netzwerk die optimalen Gewichte, indem es Fehler minimiert.
        - Dies geschieht durch einen Prozess namens **Backpropagation** und **Gradientenabstieg**.
    
    ---
    
    ## Arten von Neuronalen Netzwerken
    
    1. **Feedforward-Netzwerke:**
        - Einfachste Form, Daten fließen in eine Richtung.
    2. **Convolutional Neural Networks (CNNs):**
        - Für Bildverarbeitung (z. B. Gesichtserkennung).
    3. **Recurrent Neural Networks (RNNs):**
        - Für sequenzielle Daten (z. B. Text, Sprache).
    4. **Generative Adversarial Networks (GANs):**
        - Erstellen neuer Daten (z. B. Bilder generieren).
    
    ---
    
    ## Anwendung
    
    - **Bild- und Spracherkennung** (z. B. Gesichtserkennung, Sprachassistenten).
    - **Medizin** (z. B. Diagnose von Krankheiten).
    - **Autonomes Fahren** (z. B. Objekterkennung).
    - **Finanzen** (z. B. Betrugserkennung).
    
    ---
    
    ## Vorteile
    
    - Automatische Merkmalsextraktion.
    - Hohe Genauigkeit bei komplexen Aufgaben.
    
    ## Herausforderungen
    
    - Benötigen große Datenmengen.
    - Lange Trainingszeiten.
    - "Black Box"-Charakter: schwer interpretierbar.
    
    ---
    
    - **Beispiel**
        
        Das `alpha`-Parameter im gezeigten Code ist der Regularisierungsparameter (L2-Regularisierung) des `MLPClassifier`. Er steuert, wie stark die Gewichte des Neuronalen Netzes während des Trainings bestraft werden, um Overfitting zu verhindern.
        
        **Ausführlicher:**
        
        - **Hintergrund**: Beim Training eines Neuronalen Netzes werden die Gewichte so angepasst, dass die Fehlerfunktion minimiert wird. Ohne Regularisierung neigt ein Modell dazu, sich stark an die Trainingsdaten anzupassen (Overfitting). Dies führt dazu, dass es auf neuen, zuvor ungesehenen Daten schlechter generalisiert.
        - **Rolle von `alpha`**:
            
            Das `alpha`-Parameter in `MLPClassifier` ist der Regularisierungsstärkefaktor für die L2-Strafterm, die auf die Gewichte angewendet wird. Es wird im Trainingsprozess in die Kostenfunktion integriert. Ist `alpha` größer, werden größere Gewichte stärker bestraft. Dadurch werden die Gewichte tendenziell kleiner gehalten, was das Modell "glatter" macht und vor Überanpassung schützen kann, aber bei zu hoher Regularisierung auch unter Umständen die Trainingsleistung verschlechtern kann.
            
        - **Effekt der Variation von `alpha`**:
            
            Der Code testet verschiedene `alpha`-Werte (logarithmisch skaliert von 0.1 bis 10). Bei kleineren Werten ist die Regularisierung schwächer, das Netzwerk kann komplexere Muster lernen, läuft aber eher Gefahr zu überfitten. Bei größeren Werten ist der Regularisierungseffekt stärker, was das Overfitting reduziert, aber eventuell die Fähigkeit des Modells einschränkt, die Trainingsdaten ausreichend genau zu modellieren.
            
        
        In Summe bestimmt `alpha` also das Gleichgewicht zwischen Overfitting (zu kleines `alpha`) und Underfitting (zu großes `alpha`) im MLPClassifier.
        
        - Code
            
            ```bash
            #!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            """
            Created on Thu Dec 12 10:17:17 2024
            
            @author: colinengelmann
            """
            
            # Authors: The scikit-learn developers
            # SPDX-License-Identifier: BSD-3-Clause
            
            import numpy as np
            from matplotlib import pyplot as plt
            from matplotlib.colors import ListedColormap
            
            from sklearn.datasets import make_circles, make_classification, make_moons
            from sklearn.model_selection import train_test_split
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            
            h = 0.02  # step size in the mesh
            
            alphas = np.logspace(-1, 1, 5)
            
            classifiers = []
            names = []
            for alpha in alphas:
                classifiers.append(
                    make_pipeline(
                        StandardScaler(),
                        MLPClassifier(
                            solver="lbfgs",
                            alpha=alpha,
                            random_state=1,
                            max_iter=2000,
                            early_stopping=True,
                            hidden_layer_sizes=[10, 10],
                        ),
                    )
                )
                names.append(f"alpha {alpha:.2f}")
            
            X, y = make_classification(
                n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1
            )
            rng = np.random.RandomState(2)
            X += 2 * rng.uniform(size=X.shape)
            linearly_separable = (X, y)
            
            datasets = [
                make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable,
            ]
            
            figure = plt.figure(figsize=(17, 9))
            i = 1
            # iterate over datasets
            for X, y in datasets:
                # split into training and test part
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.4, random_state=42
                )
            
                x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
                # just plot the dataset first
                cm = plt.cm.RdBu
                cm_bright = ListedColormap(["#FF0000", "#0000FF"])
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                # Plot the training points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
                # and testing points
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                i += 1
            
                # iterate over classifiers
                for name, clf in zip(names, classifiers):
                    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)
            
                    # Plot the decision boundary. For that, we will assign a color to each
                    # point in the mesh [x_min, x_max] x [y_min, y_max].
                    if hasattr(clf, "decision_function"):
                        Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
                    else:
                        Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]
            
                    # Put the result into a color plot
                    Z = Z.reshape(xx.shape)
                    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
            
                    # Plot also the training points
                    ax.scatter(
                        X_train[:, 0],
                        X_train[:, 1],
                        c=y_train,
                        cmap=cm_bright,
                        edgecolors="black",
                        s=25,
                    )
                    # and testing points
                    ax.scatter(
                        X_test[:, 0],
                        X_test[:, 1],
                        c=y_test,
                        cmap=cm_bright,
                        alpha=0.6,
                        edgecolors="black",
                        s=25,
                    )
            
                    ax.set_xlim(xx.min(), xx.max())
                    ax.set_ylim(yy.min(), yy.max())
                    ax.set_xticks(())
                    ax.set_yticks(())
                    ax.set_title(name)
                    ax.text(
                        xx.max() - 0.3,
                        yy.min() + 0.3,
                        f"{score:.3f}".lstrip("0"),
                        size=15,
                        horizontalalignment="right",
                    )
                    i += 1
            
            figure.subplots_adjust(left=0.02, right=0.98)
            plt.show()
            ```
            
        
        ![image.png](Maschine%20Learning%20Notizen%201585d8a99a52809eb586f58eba0e4842/image%2017.png)
        
- Feature Selection
    
    Feature Selektion ist ein kritischer Schritt bei der Datenvorbereitung für maschinelles Lernen. Es geht darum, aus einer Vielzahl vo Merkmalen nur die aussagekräftigsten auszuwählen, um die Vorhersagequalität zu verbessern und Überanpassung zu vermeiden. Ziel ist es die Komplexität des Modells zu reduzieren, die Rechenressourcen zu schonen und die Vorhersagequalität zu verbessern. Typische Schritte in der Feature Selection umfassen:
    
    - Bewertung jedes Features auf seine Relevanz und Redundanz
    - Enfernen irrelevanter oder stark korrelierter Features
    - Anwenden von Modellen zur Bewertung der Einflussnahme einzelner Features
    
    Wichtig ist es deshalb weil damit die Genauigkeit von modellen verbessert werden kann, indem es unnötige Datenfaktoren eliminiert. Gleichzeitig wird die Interpretierbarkeit von den Modellen erhöht unter einer verringerung der Trainingszeit, durch Dimensionsreduktion.
    
    - Beispiel
        
        Angenommen, Du hast einen Datensatz zur Vorhersage von Autopreisen. Wenn Du unnötige Merkmale wie Fahrzeugsfarbe entfernen und Dich auf wichtige Merkmale wie den Kilometerstand, das Baujahr und die Marke konzentrierst, wird das Modell effizienter und aussagekräftiger.
        
    
    Mathematische Grundlage der Feature Selection kann durch die Minimierung von Infomationsverlust erläutert werden. Man bestimmt die Korrelation eines Ffeatures mit dem Zielattribut. Ein hoher Korrealationswert deutet dann auf einen hohe Relevanz hin. Gängige Mathematische Beschreibung ist der F-Test:
    
    $$
    F = \frac{\text{Varianz zwischen den Gruppen}}{\text{Varianz innerhalb der Gruppen}}
    $$
    
    Ein höherer F-Wert zeigt an, dass das Feature signifikant zur Zielvariablen beiträgt.
    
    ## Methoden der Feature Selection
    
    - **Filter-Methoden:** Diese bewerten die Relevanz eines Features, ohne auf ein Modell Bezug zu nehmen. Dazu gehören die Varianzschwelle (removes low-variance features) und der Chi-Quadrat-Test.
    - **Wrapper-Methoden**: Hier wird ein Prädiktionsmodell trainiert, um die Performance zu bewerten und die besten Featurekombinationen zu identifizieren, wie in der rekursiven Merkmalselimination (RFE).
    - **Embedded-Methoden**: Diese führen während des Modelltrainings eine Feature Selection durch. Entscheidungsbäume und regulierte Modelle wie Lasso sind gängige Ansätze.
    
    - **Mutual Information**
        
        # Mutual Information in Feature Selection
        
        **Mutual Information (MI)** ist ein Konzept aus der Informationstheorie, das in der **Feature Selection** verwendet wird, um die Abhängigkeit zwischen einem Merkmal (Feature) und dem Ziel (Label) zu bewerten. Es misst, wie viel Information ein Merkmal über die Zielvariable liefert.
        
        ---
        
        ## Grundidee der Mutual Information
        
        Mutual Information quantifiziert den Grad der **Nicht-Unabhängigkeit** zwischen zwei Variablen X (Feature) und Y (Label). Wenn X und Y vollständig unabhängig sind, ist die MI gleich 0. Eine höhere MI zeigt eine stärkere Abhängigkeit zwischen den Variablen an.
        
        Die mathematische Definition lautet:
        
        $$
        
        I(X; Y) = \sum_{x \in X} \sum_{y \in Y} P(x, y) \cdot \log \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right)
        
        $$
        
        ### Begriffe:
        
        - \( P(x, y) \): Gemeinsame Wahrscheinlichkeitsverteilung von \( X \) und \( Y \).
        - \( P(x) \): Marginale Wahrscheinlichkeitsverteilung von \( X \).
        - \( P(y) \): Marginale Wahrscheinlichkeitsverteilung von \( Y \).
        
        ---
        
        ## Warum MI in der Feature Selection?
        
        In der Feature Selection hilft MI dabei, die Features auszuwählen, die die meisten Informationen über die Zielvariable enthalten. Das Ziel ist es, die Merkmale zu identifizieren, die \( Y \) am besten vorhersagen können.
        
        ---
        
        ## Schritte der Feature Selection mit Mutual Information
        
        1. **Berechnung der MI**: Für jedes Feature \( X_i \) berechnet man die Mutual Information \( I(X_i; Y) \) zwischen dem Feature und dem Ziel.
        2. **Feature Ranking**: Die Features werden basierend auf ihrem MI-Wert in absteigender Reihenfolge sortiert.
        3. **Feature Auswahl**: Man wählt die \( k \)-besten Features oder alle Features aus, die eine MI oberhalb eines bestimmten Schwellenwerts haben.
        
        ---
        
        ## Beispiel
        
        Stell dir vor, wir haben ein Datenset mit den Features:
        
        - \( X_1 \): Alter einer Person
        - \( X_2 \): Gewicht einer Person
        - \( X_3 \): Schuhgröße
        Und das Ziel: \( Y \): Wahrscheinlichkeit, ob die Person ein Athlet ist.
        
        Wenn wir die MI berechnen:
        
        - \( I(X_1; Y) = 0.7 \)
        - \( I(X_2; Y) = 0.5 \)
        - \( I(X_3; Y) = 0.1 \)
        
        **Interpretation**:
        
        - Alter (\( X_1 \)) liefert die meisten Informationen über die Zielvariable \( Y \).
        - Gewicht (\( X_2 \)) hat weniger, aber immer noch relevante Informationen.
        - Schuhgröße (\( X_3 \)) ist kaum informativ.
        
        Man würde sich bei der Feature Selection wahrscheinlich für \( X_1 \) und \( X_2 \) entscheiden.
        
        ---
        
        ## Vorteile von MI
        
        1. Kann **nicht-lineare Abhängigkeiten** erkennen, im Gegensatz zu Korrelation.
        2. Funktioniert sowohl mit **kontinuierlichen** als auch mit **diskreten** Variablen.
        
        ## Nachteile von MI
        
        1. Berechnung ist oft rechnerisch aufwendig.
        2. MI allein berücksichtigt nicht die **Redundanz** zwischen Features (z. B., wenn zwei Features dieselbe Information liefern).
        
        ---
        
    - Codebeispiel
        
        Dieses Beispiel veranschaulicht die Unterschiede zwischen der univariaten F-Test-Statistik und der gegenseitigen Information.
        
        Wir betrachten 3 Merkmale x_1, x_2, x_3, die gleichmäßig über [0, 1] verteilt sind, das Ziel hängt von ihnen wie folgt ab:
        
        y = x_1 + sin(6 * pi * x_2) + 0,1 * N(0, 1), d. h. das dritte Merkmal ist völlig irrelevant.
        
        Der folgende Code stellt die Abhängigkeit von y von den einzelnen x_i und die normalisierten Werte der univariaten F-Tests und der gegenseitigen Information dar.
        
        Da der F-Test nur die lineare Abhängigkeit erfasst, wird x_1 als das am stärksten diskriminierende Merkmal eingestuft. Andererseits kann die gegenseitige Information jede Art von Abhängigkeit zwischen Variablen erfassen und bewertet x_2 als das diskriminierendste Merkmal, was wahrscheinlich besser mit unserer intuitiven Wahrnehmung für dieses Beispiel übereinstimmt. Beide Methoden stufen x_3 korrekt als irrelevant ein.