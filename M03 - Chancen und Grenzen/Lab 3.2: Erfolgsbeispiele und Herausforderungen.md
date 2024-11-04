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

### Herausforderung 2: Modell-Interpretierbarkeit und Transparenz

- **Problem**: Einige ML-Modelle, wie neuronale Netze, sind schwer zu interpretieren. Das Fehlen von Transparenz kann die Akzeptanz von ML-Modellen verringern, insbesondere in Bereichen, in denen die Nachvollziehbarkeit entscheidend ist.

---
#### Codebeispiel: Feature Importance in Entscheidungsbäumen

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Beispiel-Datensatz
X = np.array([[1.5, 0.3], [1.8, 0.4], [2.0, 0.6], [1.3, 0.2], [1.7, 0.5], [2.1, 0.8]])
y = np.array([1, 1, 1, 0, 0, 0])  # Klassifikation

# Entscheidungsbaum-Modell erstellen und trainieren
model = DecisionTreeClassifier()
model.fit(X, y)

# Wichtigkeit der Merkmale anzeigen
feature_importances = model.feature_importances_
print("Feature Importance:", feature_importances)

```
---

### Code für **Erklärung zu Feature Importance in Entscheidungsbäumen**

```markdown
**Erklärung**:
- Die Merkmalswichtigkeit gibt an, welche Variablen am meisten Einfluss auf das Modell haben. Das hilft, die Entscheidungsgrundlage des Modells transparenter zu machen.
