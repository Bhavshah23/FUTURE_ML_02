import joblib
import re
import os

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

category_model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "category_model.pkl"
    )
)

priority_model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "priority_model.pkl"
    )
)

vectorizer = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "tfidf_vectorizer.pkl"
    )
)

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def predict_ticket(ticket):

    cleaned = clean_text(ticket)

    vector = vectorizer.transform([cleaned])

    category = category_model.predict(vector)[0]

    priority = priority_model.predict(vector)[0]

    return category, priority