# Lab 3.2: Erfolgsbeispiele und Herausforderungen – Übungsaufgabe

## Einleitung
In dieser Übung untersuchen Sie ein erfolgreiches Anwendungsbeispiel von Machine Learning und analysieren die damit verbundenen Herausforderungen. Ziel ist es, besser zu verstehen, was zum Erfolg von ML-Projekten beiträgt und welche Hürden überwunden werden müssen, um ML erfolgreich zu implementieren.

---

## Übung: Vorhersage von Kundenabwanderung im Telekommunikationsbereich

### Aufgabe
Stellen Sie sich vor, Sie arbeiten für ein Telekommunikationsunternehmen, das Machine Learning verwenden möchte, um die Abwanderung von Kunden (Churn) vorherzusagen. Ihre Aufgabe ist es, ein einfaches Modell zur Vorhersage der Kundenabwanderung zu entwickeln und dann zu reflektieren, welche Herausforderungen bei der Umsetzung eines solchen Projekts auftreten könnten.

### Schritte zur Lösung
1. Erstellen Sie einen kleinen Datensatz mit Kundenmerkmalen, z. B. `Nutzungsdauer` (in Monaten), `Monatliche Ausgaben` und `Anzahl der Support-Anrufe`.
2. Kennzeichnen Sie Kunden als `1` (Kunde wechselt den Anbieter) oder `0` (Kunde bleibt).
3. Teilen Sie die Daten in Trainings- und Testdaten auf.
4. Trainieren Sie ein Klassifikationsmodell (z. B. Logistic Regression) mit den Trainingsdaten.
5. Überprüfen Sie die Genauigkeit und reflektieren Sie anschließend, welche Herausforderungen bei der Umsetzung eines solchen Projekts auftreten könnten.

### Hilfestellung und Codebeispiel

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz: Kundenmerkmale (Nutzungsdauer, monatliche Ausgaben, Support-Anrufe)
X = np.array([[12, 50, 1], [24, 80, 3], [6, 30, 0], [48, 100, 5], [36, 70, 2], [18, 60, 1]])
y = np.array([1, 0, 1, 0, 0, 1])  # 1 = Kunde wechselt, 0 = Kunde bleibt

# Schritt 1: Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Schritt 2: Logistic Regression-Modell erstellen und trainieren
model = LogisticRegression()
model.fit(X_train, y_train)

# Schritt 3: Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)

