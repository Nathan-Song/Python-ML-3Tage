# Lab 1.3: Transferaufgabe – ML-Potenziale im eigenen Arbeitsumfeld – Übungsaufgabe

## Einleitung
In dieser Übung analysieren Sie Ihr eigenes Arbeitsumfeld, um potenzielle Anwendungen von Machine Learning (ML) zu identifizieren. Das Ziel ist es, zu verstehen, wie ML bestimmte Aufgaben und Prozesse optimieren kann. Sie entwickeln eine erste Idee für ein ML-Projekt und überlegen, welche Art von Daten und Modellen dafür erforderlich sind.

---

## Übung: Identifikation von ML-Potenzialen im eigenen Arbeitsumfeld

### Aufgabe
Überlegen Sie sich eine wiederkehrende Aufgabe oder Herausforderung in Ihrem Arbeitsumfeld, die durch Machine Learning verbessert werden könnte. Erstellen Sie eine kurze Beschreibung der Aufgabe und entwickeln Sie eine Idee für ein ML-Projekt. Folgende Fragen sollen dabei helfen:

1. **Beschreibung des Problems**: Welche Aufgabe könnte durch ML optimiert werden? Worin besteht die Herausforderung?
2. **Datenerfassung**: Welche Daten wären für die Lösung des Problems erforderlich? Sind diese Daten verfügbar?
3. **ML-Ansatz**: Welche Art von ML-Algorithmus wäre für dieses Problem geeignet? (z. B. Klassifikation, Regression, Clustering)
4. **Erwarteter Nutzen**: Welche Vorteile würden sich aus der ML-Lösung ergeben?

### Hilfestellung und Beispielidee

#### Beispielidee: Automatische Bearbeitung von Kundenanfragen

- **Problem**: Das Unternehmen erhält täglich eine große Menge an Kundenanfragen. Derzeit werden diese manuell durch den Kundenservice bearbeitet, was zeitaufwändig ist. Ein ML-Modell könnte die Anfragen automatisch kategorisieren und priorisieren.
  
- **Datenerfassung**: Für die Kategorisierung der Anfragen könnten Textdaten der Anfragen genutzt werden. Diese Daten könnten aus E-Mails oder Support-Tickets stammen.
  
- **ML-Ansatz**: Ein Klassifikationsmodell (z. B. Naive Bayes oder Decision Tree) könnte entwickelt werden, um Anfragen automatisch in Kategorien wie "Technische Unterstützung", "Abrechnung", "Allgemeine Fragen" zu klassifizieren.
  
- **Erwarteter Nutzen**: Die Bearbeitungszeit für Kundenanfragen könnte verkürzt werden, und dringende Anfragen könnten priorisiert werden. Dadurch verbessert sich die Effizienz im Kundenservice und die Kundenzufriedenheit steigt.

### Codebeispiel: Vorbereitung der Daten und Modell für die Anfrageklassifikation

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
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
categories = ["Technische Unterstützung", "Abrechnung", "Technische Unterstützung", "Technische Unterstützung", "Technische Unterstützung", "Abrechnung"]

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

