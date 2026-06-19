import joblib

from preprocess import load_data

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

df = load_data("../data/all_tickets_processed_improved_v3.csv")

X = df["cleaned_text"]
y = df["Topic_group"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

report = classification_report(y_test, predictions)

print(report)

with open("../reports/category_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(14,10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
    cbar=True
)

plt.title(
    "Support Ticket Category Classification Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Predicted Category",
    fontsize=12
)

plt.ylabel(
    "Actual Category",
    fontsize=12
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "../reports/category_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()