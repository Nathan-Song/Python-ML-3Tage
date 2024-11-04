# Lab 3.1: Möglichkeiten und Grenzen von ML-Anwendungen

## Einleitung
Machine Learning (ML) bietet große Chancen, um Prozesse zu automatisieren, Muster zu erkennen und Vorhersagen zu treffen, die für viele Bereiche von Vorteil sind. Allerdings hat ML auch Grenzen, die die Umsetzung und den Einsatz in der Praxis einschränken können. In diesem Lab werden wir die Möglichkeiten und Grenzen von ML-Anwendungen beleuchten und anhand von Beispielen verdeutlichen.

---

## 1. Möglichkeiten von ML-Anwendungen

Machine Learning ermöglicht die Lösung komplexer Probleme und bietet in vielen Branchen Potenzial für Innovation und Effizienzsteigerung. Hier sind einige der wichtigsten Möglichkeiten:

### Automatisierung und Prozessoptimierung

- **Beispiel**: Automatisierte Qualitätskontrolle in der Fertigung, bei der ML-Modelle Defekte in Produkten erkennen und damit den manuellen Prüfaufwand reduzieren.
  
#### Codebeispiel: Automatische Erkennung von Defekten mit Entscheidungsbäumen

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz: Merkmale eines Produkts (z. B. Größe, Gewicht) und Defektstatus (1 = defekt, 0 = intakt)
X = np.array([[5.0, 200], [5.5, 190], [6.0, 210], [7.5, 180], [8.0, 220], [8.5, 205]])  # Größe, Gewicht
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = intakt, 1 = defekt

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell erstellen und trainieren
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)

```
---

## 2. Grenzen von ML-Anwendungen

Trotz der vielen Vorteile stoßen ML-Anwendungen in der Praxis auch an ihre Grenzen. Einige der wesentlichen Herausforderungen und Einschränkungen sind:

### Datenqualität und -verfügbarkeit

- **Herausforderung**: ML-Modelle sind stark von der Qualität und Menge der verfügbaren Daten abhängig. Fehlende, unvollständige oder ungenaue Daten können die Modellleistung beeinträchtigen.
  
#### Codebeispiel: Datenvorbereitung und Umgang mit fehlenden Werten

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Beispiel-Datensatz mit fehlenden Werten
data = {'Größe': [5.0, 5.5, None, 7.5, 8.0, None], 'Gewicht': [200, 190, 210, None, 220, 205]}
df = pd.DataFrame(data)

# Fehlende Werte mit dem Mittelwert auffüllen
imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("Daten nach der Imputation:")
print(df_filled)

