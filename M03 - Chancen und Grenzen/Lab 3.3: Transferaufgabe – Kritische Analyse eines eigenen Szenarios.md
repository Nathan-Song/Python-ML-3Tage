# Lab 3.3: Transferaufgabe – Kritische Analyse eines eigenen Szenarios

## Einleitung
In diesem Lab geht es darum, ein eigenes Szenario zu analysieren und zu beurteilen, wie Machine Learning eingesetzt werden kann, um Herausforderungen zu bewältigen oder Prozesse zu optimieren. Ziel ist es, eine kritische Analyse durchzuführen, um potenzielle Chancen und Grenzen von ML für das gewählte Szenario zu identifizieren.

---

## 1. Szenarioauswahl und Beschreibung

Stellen Sie sich vor, Sie arbeiten in einem Unternehmen, das Kundenanfragen im Support automatisch kategorisieren und priorisieren möchte, um eine effizientere Bearbeitung sicherzustellen. Ziel ist es, Machine Learning einzusetzen, um Anfragen automatisch nach Kategorien (z. B. "Technisches Problem", "Abrechnungsfrage", "Allgemeine Frage") zu klassifizieren und die Dringlichkeit zu bewerten.

### Beispielanwendungen
- **Textklassifikation**: Kategorisierung von Kundenanfragen basierend auf den Inhalten der Anfragen.
- **Priorisierung**: Einschätzung der Dringlichkeit der Anfragen, um eine schnellere Bearbeitung wichtiger Anliegen zu ermöglichen.

### Codebeispiel: Textklassifikation für Kundenanfragen mit Naive Bayes

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Beispiel-Datensatz: Kundenanfragen und Kategorien
texts = [
    "Mein Internet ist sehr langsam",
    "Wann wird die Rechnung verschickt?",
    "Ich kann mich nicht einloggen",
    "Wie kann ich mein Passwort ändern?",
    "Warum ist mein Konto gesperrt?",
    "Ich habe eine Frage zur Abrechnung"
]
categories = ["Technisches Problem", "Abrechnungsfrage", "Technisches Problem", "Technisches Problem", "Technisches Problem", "Abrechnungsfrage"]

# Textdaten in Merkmale umwandeln
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.3, random_state=42)

# Naive Bayes-Modell erstellen und trainieren
model = MultinomialNB()
model.fit(X_train, y_train)

# Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)

```
---

## 2. Kritische Analyse: Chancen und Herausforderungen

Die Implementierung von ML zur automatischen Klassifikation und Priorisierung von Kundenanfragen bietet Potenzial, aber auch Herausforderungen:

### Chancen

- **Effizienzsteigerung**: Automatisierte Klassifikation und Priorisierung beschleunigt die Bearbeitung von Anfragen und reduziert den manuellen Aufwand.
- **Kundenzufriedenheit**: Wichtige Anliegen können schneller bearbeitet werden, was die Kundenzufriedenheit erhöht.

### Herausforderungen

- **Datenqualität**: Die Qualität des Modells hängt stark von der Datenqualität ab. Unvollständige oder nicht repräsentative Anfragedaten können zu ungenauen Klassifikationen führen.
  
#### Codebeispiel: Umgang mit fehlenden Werten

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Beispiel-Datensatz mit fehlenden Werten
data = {'Anfrage': ["Internet langsam", None, "Rechnung unklar", "Konto gesperrt", None]}
df = pd.DataFrame(data)

# Fehlende Werte durch "Unbekannt" ersetzen
imputer = SimpleImputer(strategy='constant', fill_value="Unbekannt")
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("Daten nach der Imputation:")
print(df_filled)

```
---
#### Codebeispiel: Umgang mit fehlenden Werten

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Beispiel-Datensatz mit fehlenden Werten
data = {'Anfrage': ["Internet langsam", None, "Rechnung unklar", "Konto gesperrt", None]}
df = pd.DataFrame(data)

# Fehlende Werte durch "Unbekannt" ersetzen
imputer = SimpleImputer(strategy='constant', fill_value="Unbekannt")
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("Daten nach der Imputation:")
print(df_filled)

```
---


---

#### Codebeispiel: Feature Importance in einem Entscheidungsbaum für Verständnis

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Beispiel-Datensatz: einfache Merkmale zur Priorisierung (z. B. Anzahl Keywords, Dringlichkeit)
X = np.array([[2, 3], [1, 2], [3, 1], [2, 5], [1, 3]])
y = np.array([1, 0, 1, 1, 0])  # 1 = Hohe Priorität, 0 = Niedrige Priorität

# Entscheidungsbaum-Modell erstellen und trainieren
model = DecisionTreeClassifier()
model.fit(X, y)

# Wichtigkeit der Merkmale anzeigen
feature_importances = model.feature_importances_
print("Feature Importance:", feature_importances)

```
---
