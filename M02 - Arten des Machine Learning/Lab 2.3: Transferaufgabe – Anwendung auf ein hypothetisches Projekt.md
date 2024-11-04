# Lab 2.3: Transferaufgabe – Anwendung auf ein hypothetisches Projekt

## Einleitung
In dieser Transferaufgabe geht es darum, das Gelernte auf ein hypothetisches Projekt anzuwenden und zu überlegen, welche Algorithmen für spezifische Aufgaben sinnvoll wären.

---

## 1. Projektbeschreibung

### Szenario
Stellen Sie sich vor, Sie arbeiten für ein E-Commerce-Unternehmen, das Kunden personalisierte Produktempfehlungen geben möchte. Ziel ist es, das Kundenerlebnis zu verbessern, indem relevante Produkte basierend auf dem bisherigen Verhalten und den Vorlieben der Kunden empfohlen werden.

### Zielsetzung
Das Ziel des Projekts ist es, ein Machine-Learning-Modell zu entwickeln, das für jeden Kunden eine Liste personalisierter Empfehlungen erstellt. Dazu sollen verschiedene Machine-Learning-Algorithmen auf spezifische Teilprobleme angewendet werden.

---

## 2. Anforderungen und Algorithmen

### Anforderungen
Für das Projekt gibt es zwei zentrale Anforderungen:
1. **Kundensegmentierung**: Identifikation von Kundengruppen mit ähnlichem Verhalten, um passende Produktempfehlungen für jede Gruppe zu entwickeln.
2. **Vorhersage von Kundeninteressen**: Analyse der Kaufhistorie und des Browsing-Verhaltens jedes Kunden, um individuelle Empfehlungen zu generieren.

### Passende Algorithmen
Für diese Anforderungen könnten folgende Algorithmen nützlich sein:

1. **Kundensegmentierung**:
   - **Algorithmus**: K-Means Clustering
   - **Grund**: K-Means ist ein unüberwachter Lernalgorithmus, der gut für die Gruppierung von Daten geeignet ist. Er kann Kunden auf Basis ihrer Kaufhistorie in Cluster einteilen.

2. **Vorhersage von Kundeninteressen**:
   - **Algorithmus**: K-Nearest Neighbors (KNN) oder Entscheidungsbaum
   - **Grund**: KNN könnte genutzt werden, um Produkte basierend auf dem Verhalten ähnlicher Kunden zu empfehlen. Alternativ könnte ein Entscheidungsbaum verwendet werden, um spezifische Vorlieben und Muster aus den Daten abzuleiten.

---

## 3. Codebeispiele für das hypothetische Projekt

In diesem Abschnitt werden Codebeispiele für die ausgewählten Algorithmen vorgestellt.

### 1. Kundensegmentierung mit K-Means Clustering

```python
from sklearn.cluster import KMeans
import numpy as np

# Beispiel-Datensatz: Kaufverhalten von Kunden (Anzahl Käufe, Durchschnittspreis)
X = np.array([[5, 20], [6, 25], [10, 40], [15, 80], [20, 100], [25, 120]])

# KMeans Modell mit 2 Clustern erstellen
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Cluster-Zuordnungen und Clusterzentren ausgeben
print("Cluster-Zuordnungen:", kmeans.labels_)
print("Cluster-Zentren:", kmeans.cluster_centers_)

