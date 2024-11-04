# Lab 1.2: Branchenüberblick – Anwendungen von ML – Übungsaufgabe

## Einleitung
In dieser Übung betrachten wir Anwendungen von Machine Learning in verschiedenen Branchen und analysieren, wie ML für konkrete Anwendungsfälle eingesetzt wird. Ziel ist es, ein besseres Verständnis dafür zu entwickeln, wie ML in der Praxis genutzt wird und welche Modelle für bestimmte Branchen geeignet sind.

---

## Übung: ML-Anwendungen in der Praxis analysieren

### Aufgabe
Angenommen, Sie arbeiten in einer Bank, die Machine Learning zur Vorhersage der Kreditwürdigkeit von Kunden einsetzen möchte. Ihre Aufgabe ist es, ein Modell zu erstellen, das basierend auf dem Kundenprofil vorhersagt, ob ein Kunde als kreditwürdig eingestuft wird oder nicht. Verwenden Sie hierfür den Decision Tree-Algorithmus.

### Schritte zur Lösung
1. Erstellen Sie einen Datensatz mit Kundenprofilen, z. B. Einkommen, Alter und Anzahl bestehender Kredite.
2. Kennzeichnen Sie jeden Kunden als `1` (kreditwürdig) oder `0` (nicht kreditwürdig).
3. Teilen Sie die Daten in Trainings- und Testdaten auf.
4. Trainieren Sie ein Decision Tree-Modell mit den Trainingsdaten.
5. Testen Sie die Genauigkeit des Modells auf den Testdaten und interpretieren Sie die Ergebnisse.

### Hilfestellung und Codebeispiel

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz: Kundenprofile (Einkommen, Alter, Anzahl bestehender Kredite)
X = np.array([[50000, 25, 1], [80000, 45, 2], [60000, 35, 1], [30000, 23, 0], [40000, 30, 3], [90000, 50, 2]])
y = np.array([1, 1, 1, 0, 0, 1])  # 1 = kreditwürdig, 0 = nicht kreditwürdig

# Schritt 1: Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Schritt 2: Decision Tree-Modell erstellen und trainieren
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Schritt 3: Vorhersagen auf den Testdaten und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)

