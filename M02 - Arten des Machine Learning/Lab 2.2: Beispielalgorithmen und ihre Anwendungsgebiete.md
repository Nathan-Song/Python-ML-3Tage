# Lab 2.2: Beispielalgorithmen und ihre Anwendungsgebiete

## Einleitung
Es gibt eine Vielzahl von Algorithmen im Bereich des Machine Learning, die für unterschiedliche Aufgaben und Anwendungsgebiete geeignet sind. In diesem Lab schauen wir uns einige der am häufigsten verwendeten Algorithmen an und untersuchen ihre typischen Anwendungsfälle.

---

## 1. Entscheidungsbaum (Decision Tree)

Ein Entscheidungsbaum ist ein baumbasiertes Modell, das Entscheidungen auf Grundlage der Merkmale eines Datensatzes trifft. Jeder Knoten im Baum repräsentiert eine Entscheidung auf Basis eines Merkmals, und die Zweige stellen mögliche Ergebnisse dieser Entscheidung dar.

### Beispielanwendungen
- **Klassifikation**: Vorhersage, ob eine E-Mail Spam ist oder nicht.
- **Regression**: Vorhersage von Hauspreisen basierend auf Eigenschaften wie Größe, Lage und Alter des Hauses.

### Codebeispiel: Entscheidungsbaum für die Klassifikation

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz (Alter, Einkommen in Tausend Euro)
X = np.array([[25, 30], [35, 40], [45, 60], [20, 20], [55, 80], [40, 50]])  # Alter, Einkommen
y = np.array([0, 0, 1, 0, 1, 1])  # 0 = Kein Kauf, 1 = Kauf

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell erstellen und trainieren
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```
---

# Lab 2.2: Beispielalgorithmen und ihre Anwendungsgebiete - Entscheidungsbaum

## Einleitung
Ein Entscheidungsbaum ist ein baumbasiertes Modell, das Entscheidungen auf Grundlage der Merkmale eines Datensatzes trifft. Jeder Knoten im Baum repräsentiert eine Entscheidung auf Basis eines Merkmals, und die Zweige stellen mögliche Ergebnisse dieser Entscheidung dar.

## Beispielanwendungen
- **Klassifikation**: Vorhersage, ob eine E-Mail Spam ist oder nicht.
- **Regression**: Vorhersage von Hauspreisen basierend auf Eigenschaften wie Größe, Lage und Alter des Hauses.

## Codebeispiel: Entscheidungsbaum für die Klassifikation

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz (Alter, Einkommen in Tausend Euro)
X = np.array([[25, 30], [35, 40], [45, 60], [20, 20], [55, 80], [40, 50]])  # Alter, Einkommen
y = np.array([0, 0, 1, 0, 1, 1])  # 0 = Kein Kauf, 1 = Kauf

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell erstellen und trainieren
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```
---

# Lab 2.2: Beispielalgorithmen und ihre Anwendungsgebiete - K-Nearest Neighbors (KNN)

## Einleitung
K-Nearest Neighbors ist ein einfacher, aber leistungsstarker Algorithmus, der Datenpunkte anhand der K nächsten Nachbarn klassifiziert oder regrediert. Das Modell selbst "lernt" nicht, sondern speichert einfach alle Datenpunkte und berechnet den Abstand zu den Nachbarn während der Vorhersage.

## Beispielanwendungen
- **Klassifikation**: Bilderkennung, z. B. zur Klassifizierung von handgeschriebenen Ziffern.
- **Regression**: Vorhersage von Immobilienpreisen auf Basis ähnlicher Immobilien in der Nachbarschaft.

## Codebeispiel: K-Nearest Neighbors zur Klassifikation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz (Punktzahl in zwei Kategorien)
X = np.array([[5, 2], [6, 3], [7, 3], [10, 10], [9, 8], [10, 8]])  # Punktzahl
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = Kategorie A, 1 = Kategorie B

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell erstellen und trainieren
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
