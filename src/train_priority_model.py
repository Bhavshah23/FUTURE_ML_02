import joblib

from preprocess import load_data

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

df = load_data("../data/all_tickets_processed_improved_v3.csv")

X = df["cleaned_text"]
y = df["Priority"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

report = classification_report(y_test, predictions)

print(report)

with open("../reports/priority_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("Priority Confusion Matrix")

plt.savefig("../reports/priority_confusion_matrix.png")

joblib.dump(model, "../models/priority_model.pkl")

print("Priority Model Saved")