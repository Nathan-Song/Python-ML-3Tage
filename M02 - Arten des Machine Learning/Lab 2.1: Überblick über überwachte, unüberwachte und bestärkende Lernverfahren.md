# Lab 2.1: Überblick über überwachte, unüberwachte und bestärkende Lernverfahren

## Einleitung
Machine Learning lässt sich in verschiedene Kategorien einteilen, die auf unterschiedlichen Lernmethoden basieren. Die drei Hauptkategorien sind:
1. **Überwachtes Lernen (Supervised Learning)**
2. **Unüberwachtes Lernen (Unsupervised Learning)**
3. **Bestärkendes Lernen (Reinforcement Learning)**

In diesem Lab werfen wir einen genaueren Blick auf diese Kategorien, ihre Anwendungsfälle und einfache Codebeispiele, um die Konzepte besser zu verstehen.

---

## 1. Überwachtes Lernen (Supervised Learning)

Beim überwachten Lernen wird das Modell mit einem gelabelten Datensatz trainiert, das heißt, die Eingabedaten sind mit den korrekten Ausgabewerten versehen. Das Ziel des Modells ist es, eine Funktion zu erlernen, die die Eingaben auf die richtigen Ausgaben abbildet.

### Beispielanwendung
- **Klassifikation**: Spam-Erkennung in E-Mails.
- **Regression**: Vorhersage von Hauspreisen.

### Codebeispiel: Lineare Regression für Hauspreisvorhersage

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Beispiel-Datensatz (Hausgröße in m² und Preis in Tausend Euro)
X = np.array([[50], [60], [80], [100], [120], [150]])  # Hausgröße in m²
y = np.array([100, 120, 160, 200, 240, 300])  # Preis in Tausend Euro

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell erstellen und trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = model.predict(X_test)

# Modellbewertung
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


## 2. Unüberwachtes Lernen (Unsupervised Learning)

Beim unüberwachten Lernen arbeitet das Modell mit nicht gelabelten Daten und versucht, Strukturen oder Muster in den Daten zu erkennen. Es gibt keine "richtigen" Antworten, sondern das Modell gruppiert oder strukturiert die Daten eigenständig.

### Beispielanwendungen
- **Clustering**: Kundensegmentierung in Marketing – Kunden werden basierend auf ihrem Verhalten in Gruppen eingeteilt.
- **Dimensionsreduktion**: Datenvisualisierung und -vorverarbeitung – das Modell reduziert die Anzahl der Variablen, während es die wichtigsten Merkmale beibehält.

### Codebeispiel: K-Means Clustering für Kundensegmentierung

```python
from sklearn.cluster import KMeans
import numpy as np

# Beispiel-Datensatz (Kaufverhalten von Kunden: Ausgaben und Häufigkeit)
X = np.array([[100, 20], [200, 30], [300, 60], [400, 80], [500, 100], [600, 120]])

# KMeans Modell mit 2 Clustern erstellen
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Cluster-Zuordnungen und Clusterzentren ausgeben
print("Cluster-Zuordnungen:", kmeans.labels_)
print("Cluster-Zentren:", kmeans.cluster_centers_)
