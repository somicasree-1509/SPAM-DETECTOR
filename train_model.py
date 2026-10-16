"""
train_model.py
Trains a spam detector using TF-IDF + Multinomial Naive Bayes
and saves it to spam_model.joblib.

Run with:  python train_model.py
"""

import sys
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_FILE = "spam.csv"
MODEL_FILE = "spam_model.joblib"


def load_dataset(path):
    """Load the CSV and return a clean DataFrame with 'label' and 'message' columns."""
    try:
        # latin-1 also works for the original Kaggle "SMS Spam Collection" file
        data = pd.read_csv(path, encoding="latin-1")
    except FileNotFoundError:
        print(f"Error: '{path}' was not found. Put it in the same folder as this script.")
        sys.exit(1)

    # The original Kaggle file uses columns v1 (label) and v2 (text)
    if "v1" in data.columns and "v2" in data.columns:
        data = data.rename(columns={"v1": "label", "v2": "message"})

    if "label" not in data.columns or "message" not in data.columns:
        print("Error: spam.csv must have 'label' and 'message' columns.")
        sys.exit(1)

    data = data[["label", "message"]].dropna()
    data["label"] = data["label"].str.strip().str.lower()
    data = data[data["label"].isin(["spam", "ham"])]
    return data


def main():
    data = load_dataset(DATASET_FILE)
    print(f"Loaded {len(data)} messages ({(data['label'] == 'spam').sum()} spam, "
          f"{(data['label'] == 'ham').sum()} ham)")

    # Convert labels to numbers: spam = 1, ham = 0
    y = (data["label"] == "spam").astype(int)
    X = data["message"]

    # Keep 20% of the data aside to check how well the model works
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: TF-IDF turns text into numbers, Naive Bayes learns from them
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words="english"),
        MultinomialNB(),
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy on test data: {accuracy:.2%}")

    # Retrain on ALL data so the final model learns from every example
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to '{MODEL_FILE}'")


if __name__ == "__main__":
    main()
