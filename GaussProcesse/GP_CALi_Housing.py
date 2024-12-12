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
