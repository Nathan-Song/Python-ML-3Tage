# Lab 1.1: Grundlagen und Konzepte von Machine Learning – Übungsaufgaben

## Einleitung
In diesen Übungsaufgaben vertiefen Sie die Grundlagen des Machine Learning, indem Sie einfache Konzepte und Methoden anwenden. Jede Übung enthält eine Anleitung und Beispielcode, der Ihnen bei der Umsetzung hilft.

---

## Übung 1: Datenvorbereitung und lineare Regression

### Aufgabe
Gegeben ist ein Datensatz mit zwei Variablen: `Fläche` (in m²) und `Preis` (in Tausend Euro) von verschiedenen Wohnungen. Ihre Aufgabe ist es, ein Modell zu erstellen, das auf Basis der Wohnfläche den Preis vorhersagt. Verwenden Sie eine lineare Regression, um das Modell zu trainieren.

### Schritte zur Lösung
1. Erstellen Sie einen Datensatz mit den Wohnflächen und Preisen.
2. Teilen Sie die Daten in Trainings- und Testdaten auf.
3. Trainieren Sie ein Modell zur linearen Regression auf den Trainingsdaten.
4. Überprüfen Sie die Genauigkeit des Modells auf den Testdaten.

### Hilfestellung und Codebeispiel

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Beispiel-Datensatz: Wohnfläche in m² und Preis in Tausend Euro
X = np.array([[50], [60], [80], [100], [120], [150]])  # Wohnfläche in m²
y = np.array([100, 120, 160, 200, 240, 300])  # Preis in Tausend Euro

# Schritt 1: Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Schritt 2: Modell erstellen und trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Schritt 3: Vorhersagen auf den Testdaten und Fehlerberechnung
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

