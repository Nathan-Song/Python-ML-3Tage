# Lab 3.1: Möglichkeiten und Grenzen von ML-Anwendungen – Übungsaufgabe

## Einleitung
In dieser Übung geht es darum, die praktischen Möglichkeiten und Grenzen von Machine Learning zu analysieren. Sie erfahren, wie ML zur Lösung bestimmter Probleme eingesetzt werden kann und welche Einschränkungen zu beachten sind. Ziel ist es, die Chancen und Herausforderungen besser zu verstehen und kritisch zu hinterfragen.

---

## Übung: Erkennen der Grenzen von Machine Learning im Kundenservice

### Aufgabe
Stellen Sie sich vor, Sie arbeiten in einem Kundenservice-Team, das Machine Learning zur Analyse von Kundenfeedback einsetzen möchte. Die Aufgabe des Modells ist es, automatisch zu erkennen, ob das Feedback positiv oder negativ ist. Entwickeln Sie ein einfaches Modell, das dies ermöglicht, und reflektieren Sie anschließend die Grenzen dieses Modells.

### Schritte zur Lösung
1. Erstellen Sie einen Datensatz mit verschiedenen Kundenkommentaren, die als positiv (`1`) oder negativ (`0`) gekennzeichnet sind.
2. Teilen Sie die Daten in Trainings- und Testdaten auf.
3. Trainieren Sie ein Modell zur Textklassifikation (z. B. Naive Bayes).
4. Überprüfen Sie die Genauigkeit und überlegen Sie, welche Grenzen dieses Modell haben könnte, z. B. bei ironischem Feedback oder komplexen Aussagen.

### Hilfestellung und Codebeispiel

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Beispiel-Datensatz: Kundenfeedback und Klassifikation (1 = positiv, 0 = negativ)
comments = [
    "Sehr zufrieden mit dem Service!",
    "Wirklich schlecht, nie wieder!",
    "Das Produkt war toll.",
    "Leider war der Support wenig hilfreich.",
    "Alles super, gerne wieder!",
    "Ich bin unzufrieden mit meiner Bestellung."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = positiv, 0 = negativ

# Textdaten in Merkmale umwandeln
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comments)

# Schritt 1: Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Schritt 2: Naive Bayes-Modell erstellen und trainieren
model = MultinomialNB()
model.fit(X_train, y_train)

# Schritt 3: Vorhersagen treffen und Genauigkeit prüfen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)

