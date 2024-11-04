# Lab 2.2: Beispielalgorithmen und ihre Anwendungsgebiete – Übungsaufgabe

## Einleitung
In dieser Übung geht es darum, einen grundlegenden Machine-Learning-Algorithmus kennenzulernen und anzuwenden. Sie verwenden den **K-Nearest Neighbors (KNN)**-Algorithmus, um eine Klassifikation durchzuführen und zu sehen, wie ML-Algorithmen eingesetzt werden können, um Daten zu kategorisieren.

---

## Übung: Klassifikation von Blumenarten mit K-Nearest Neighbors (KNN)

### Aufgabe
Sie arbeiten mit einem einfachen Datensatz, der Blumenarten anhand von Blütenmerkmalen (z. B. Blütenblattlänge und -breite) klassifiziert. Ihre Aufgabe ist es, den KNN-Algorithmus zu verwenden, um festzustellen, zu welcher Art eine neue Blume gehört.

### Schritte zur Lösung
1. Erstellen Sie einen Datensatz mit Blütenmerkmalen (Länge und Breite) und der jeweiligen Klassifikation (z. B. `0` für Art A und `1` für Art B).
2. Teilen Sie die Daten in Trainings- und Testdaten auf.
3. Trainieren Sie ein K-Nearest Neighbors-Modell (KNN) mit den Trainingsdaten.
4. Testen Sie das Modell auf den Testdaten und berechnen Sie die Genauigkeit.

### Hilfestellung und Codebeispiel

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz: Blütenmerkmale (Blütenblattlänge und -breite) und Klassifikation (0 = Art A, 1 = Art B)
X = np.array([[1.5, 0.2], [1.6, 0.3], [5.0, 1.5], [5.2, 1.6], [4.9, 1.5], [1.4, 0.2]])
y = np.array([0, 0, 1, 1, 1, 0])  # 0 = Art A, 1 = Art B

# Schritt 1: Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Schritt 2: KNN-Modell erstellen und trainieren
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Schritt 3: Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)

