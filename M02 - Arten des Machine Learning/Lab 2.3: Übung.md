# Lab 2.3: Transferaufgabe – Anwendung auf ein hypothetisches Projekt – Übungsaufgabe

## Einleitung
In dieser Übung setzen Sie Machine Learning auf ein hypothetisches Szenario an und entwickeln ein einfaches Modell, das bei der Lösung eines Problems unterstützen soll. Ziel ist es, eine Vorstellung davon zu bekommen, wie ML in realen Projekten eingesetzt wird und wie Modelle erstellt und getestet werden können.

---

## Übung: Vorhersage der Kundenzufriedenheit im Kundenservice

### Aufgabe
Stellen Sie sich vor, Sie arbeiten in einem Kundenservice-Unternehmen und möchten ein Modell entwickeln, das vorhersagt, ob ein Kunde nach einem Anruf zufrieden oder unzufrieden sein wird. Sie verwenden die Länge des Gesprächs (in Minuten) und die Anzahl der Eskalationen (z. B. das Weiterleiten an einen Vorgesetzten) als Merkmale.

### Schritte zur Lösung
1. Erstellen Sie einen kleinen Datensatz mit den Merkmalen `Gesprächsdauer` und `Anzahl Eskalationen` sowie dem Ergebnis `zufrieden` oder `unzufrieden`.
2. Teilen Sie die Daten in Trainings- und Testdaten auf.
3. Trainieren Sie ein Modell zur Klassifikation (z. B. Decision Tree) mit den Trainingsdaten.
4. Testen Sie das Modell auf den Testdaten und berechnen Sie die Genauigkeit.

### Hilfestellung und Codebeispiel

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz: Gesprächsdauer (Minuten), Anzahl Eskalationen, Klassifikation (1 = zufrieden, 0 = unzufrieden)
X = np.array([[5, 0], [10, 1], [3, 0], [8, 2], [12, 3], [6, 1], [4, 0], [9, 2]])
y = np.array([1, 1, 1, 0, 0, 1, 1, 0])  # 1 = zufrieden, 0 = unzufrieden

# Schritt 1: Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Schritt 2: Decision Tree-Modell erstellen und trainieren
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Schritt 3: Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)

