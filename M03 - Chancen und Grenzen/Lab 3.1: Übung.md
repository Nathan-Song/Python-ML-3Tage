# Lab 3.1: Möglichkeiten und Grenzen von ML-Anwendungen – Übungsaufgabe

## Einleitung
In dieser Übung befassen Sie sich mit den praktischen Grenzen von Machine Learning, insbesondere im Bereich der Bilderkennung. Sie erfahren, welche Herausforderungen bei der Bilderkennung auftreten können, und lernen, die Leistungsfähigkeit und Grenzen eines einfachen Modells zu bewerten.

---

## Übung: Bilderkennung zur Defekterkennung in der Fertigung

### Aufgabe
Stellen Sie sich vor, Sie arbeiten in einem Produktionsunternehmen, das Machine Learning zur Erkennung defekter Produkte einsetzen möchte. Ihre Aufgabe ist es, ein einfaches Modell zu entwickeln, das auf Basis von Bildmerkmalen erkennt, ob ein Produkt defekt oder intakt ist. Anschließend reflektieren Sie über die Grenzen dieses Modells und diskutieren, welche Herausforderungen dabei auftreten können.

### Schritte zur Lösung
1. Erstellen Sie einen kleinen Datensatz mit Merkmalen, die Bildinformationen simulieren (z. B. Helligkeit und Kontrast).
2. Kennzeichnen Sie die Daten als `0` (intakt) oder `1` (defekt).
3. Teilen Sie die Daten in Trainings- und Testdaten auf.
4. Trainieren Sie ein Decision Tree-Modell zur Klassifikation.
5. Testen Sie das Modell auf den Testdaten und reflektieren Sie anschließend über die Grenzen und Herausforderungen dieses Ansatzes.

### Hilfestellung und Codebeispiel

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz: Merkmale zur Simulation von Bildinformationen (Helligkeit, Kontrast)
X = np.array([[200, 50], [180, 45], [120, 30], [150, 35], [100, 25], [190, 40]])
y = np.array([0, 0, 1, 1, 1, 0])  # 0 = intakt, 1 = defekt

# Schritt 1: Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Schritt 2: Decision Tree-Modell erstellen und trainieren
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Schritt 3: Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)
