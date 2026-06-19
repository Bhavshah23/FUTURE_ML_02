import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def load_data(path):

    df = pd.read_csv(path)

    priority_mapping = {
        "Access": "High",
        "Administrative rights": "High",
        "Hardware": "Medium",
        "Storage": "Medium",
        "Purchase": "Medium",
        "HR Support": "Low",
        "Internal Project": "Low",
        "Miscellaneous": "Low"
    }

    df["Priority"] = df["Topic_group"].map(priority_mapping)

    df["cleaned_text"] = df["Document"].apply(clean_text)

    return df