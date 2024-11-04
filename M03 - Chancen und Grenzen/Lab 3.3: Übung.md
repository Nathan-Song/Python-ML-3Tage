# Lab 3.3: Transferaufgabe – Kritische Analyse eines eigenen Szenarios – Übungsaufgabe

## Einleitung
In dieser Übung analysieren Sie ein Szenario aus Ihrem eigenen oder einem hypothetischen Arbeitsumfeld und beurteilen, wie Machine Learning bei der Lösung eines spezifischen Problems unterstützen könnte. Die Aufgabe fördert ein kritisches Verständnis für die Grenzen und Möglichkeiten von ML und regt zur Reflexion über die praktische Umsetzung an.

---

## Übung: Bestandsprognose im Einzelhandel

### Aufgabe
Stellen Sie sich vor, Sie arbeiten im Einzelhandel und möchten die Bestände Ihrer Filialen optimieren, um Überbestände und Engpässe zu vermeiden. Ihre Aufgabe ist es, ein einfaches ML-Modell zu entwickeln, das den wöchentlichen Bedarf eines Produkts vorhersagt. Nach der Modellierung reflektieren Sie, welche Herausforderungen bei der Umsetzung auftreten könnten.

### Schritte zur Lösung
1. Erstellen Sie einen kleinen Datensatz mit historischen Verkaufsdaten eines Produkts, z. B. `Verkaufsmenge`, `Preisänderungen` und `Saisonale Einflüsse`.
2. Verwenden Sie die Daten, um ein Regressionsmodell zu trainieren, das die Verkaufsmenge in der nächsten Woche vorhersagt.
3. Teilen Sie die Daten in Trainings- und Testdaten auf und trainieren Sie das Modell.
4. Überprüfen Sie die Modellleistung und reflektieren Sie über die Grenzen und Herausforderungen eines solchen Bestandsprognose-Systems.

### Hilfestellung und Codebeispiel

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Beispiel-Datensatz: historische Verkaufsdaten (Verkaufsmenge, Preisänderungen, Saisonale Einflüsse)
X = np.array([[200, 0, 1], [250, 1, 0], [150, -1, 1], [300, 0, 0], [220, 1, 1], [170, -1, 0]])
y = np.array([210, 240, 160, 290, 225, 180])  # Verkaufsmenge in der folgenden Woche

# Schritt 1: Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Schritt 2: Lineares Regressionsmodell erstellen und trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Schritt 3: Vorhersagen treffen und Fehler berechnen
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

