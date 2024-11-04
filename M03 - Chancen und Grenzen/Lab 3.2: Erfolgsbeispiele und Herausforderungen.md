# Lab 3.2: Erfolgsbeispiele und Herausforderungen

## Einleitung
Machine Learning (ML) hat in vielen Bereichen bereits große Erfolge erzielt und ermöglicht innovative Lösungen, die zuvor nicht denkbar waren. Gleichzeitig bringt die Implementierung von ML-Projekten auch Herausforderungen mit sich. In diesem Lab werden einige erfolgreiche Anwendungsbeispiele von ML betrachtet sowie typische Herausforderungen, die bei der Umsetzung auftreten.

---

## 1. Erfolgsbeispiele von ML-Anwendungen

Machine Learning hat sich in vielen Industrien als wertvolles Werkzeug erwiesen. Hier sind einige erfolgreiche Anwendungsfälle:

### Bild- und Spracherkennung

- **Beispiel**: Gesichtserkennungssysteme, wie sie von Smartphones zur Entsperrung verwendet werden.
- **Erfolg**: Gesichtserkennungsalgorithmen haben eine hohe Genauigkeit erreicht, die das Nutzererlebnis verbessert und gleichzeitig die Sicherheit erhöht.

#### Codebeispiel: Bildklassifizierung mit einem einfachen Entscheidungsbaum

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Beispiel-Datensatz: einfache Merkmale von Bildern (z. B. Helligkeit, Kontrast) und Kategorie (0 = Tier, 1 = Objekt)
X = np.array([[0.6, 0.4], [0.8, 0.3], [0.2, 0.7], [0.9, 0.5], [0.3, 0.9], [0.5, 0.8]])
y = np.array([0, 1, 0, 1, 0, 0])  # 0 = Tier, 1 = Objekt

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
