# Lab 2.1: Überblick über überwachte, unüberwachte und bestärkende Lernverfahren – Übungsaufgabe

## Einleitung
In dieser Übung geht es darum, ein grundlegendes Verständnis für die drei Hauptarten des Machine Learnings zu entwickeln: Überwachtes Lernen, unüberwachtes Lernen und bestärkendes Lernen. Sie lernen, wie einfache Modelle für jede Lernmethode erstellt werden und wie sie sich in der Praxis unterscheiden.

---

## Übung: Grundlagen der Lernverfahren verstehen

### Aufgabe
In dieser Aufgabe probieren Sie ein einfaches Modell für **überwachtes** und **unüberwachtes Lernen** aus. Ziel ist es, die Unterschiede und Funktionsweisen der beiden Lernverfahren kennenzulernen.

1. **Überwachtes Lernen**: Erstellen Sie ein Modell, das auf Basis der Körpergröße und Schuhgröße vorhersagt, ob eine Person ein Erwachsener oder ein Kind ist.
2. **Unüberwachtes Lernen**: Gruppieren Sie verschiedene Punkte basierend auf deren Position (Clustering), ohne dass es vorgegebene Kategorien gibt.

### 1. Überwachtes Lernen – Klassifikation mit Decision Tree

Hier lernen Sie, wie überwachtes Lernen funktioniert, indem Sie ein Decision Tree-Modell verwenden, das zwischen Kindern und Erwachsenen unterscheidet.

#### Schritte zur Lösung
1. Erstellen Sie einen kleinen Datensatz mit Körpergröße, Schuhgröße und der Klassifikation als `0` (Kind) oder `1` (Erwachsener).
2. Trainieren Sie ein Decision Tree-Modell mit diesen Daten.
3. Verwenden Sie das Modell, um eine Vorhersage zu treffen.

#### Hilfestellung und Codebeispiel

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz: Körpergröße in cm, Schuhgröße und Klassifikation (0 = Kind, 1 = Erwachsener)
X = np.array([[120, 30], [130, 32], [180, 42], [160, 38], [155, 37], [140, 34]])
y = np.array([0, 0, 1, 1, 1, 0])  # 0 = Kind, 1 = Erwachsener

# Schritt 1: Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Schritt 2: Decision Tree-Modell erstellen und trainieren
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Schritt 3: Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)

```
---


### Hilfestellung für Unüberwachtes Lernen – Clustering mit K-Means
--

#### Schritte zur Lösung

1. Erstellen Sie einen kleinen Datensatz mit Koordinatenpunkten.
2. Verwenden Sie den K-Means-Algorithmus, um diese Punkte in zwei Gruppen zu unterteilen.

#### Hilfestellung und Codebeispiel

```python
from sklearn.cluster import KMeans
import numpy as np

# Beispiel-Datensatz: Koordinatenpunkte (x, y)
X = np.array([[1, 2], [2, 3], [3, 3], [8, 8], [9, 10], [10, 8]])

# KMeans-Modell mit 2 Clustern erstellen
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Cluster-Zuordnungen und Clusterzentren ausgeben
print("Cluster-Zuordnungen:", kmeans.labels_)
print("Cluster-Zentren:", kmeans.cluster_centers_)
