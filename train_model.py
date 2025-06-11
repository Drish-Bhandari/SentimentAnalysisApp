import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import re

# Load datasets
def load_data():
    imdb = pd.read_csv("IMDB Dataset.csv")
    alexa = pd.read_csv("amazon_alexa_reviews.csv")

    # Standardize columns
    imdb = imdb.rename(columns={"review": "text", "sentiment": "label"})
    imdb["label"] = imdb["label"].map({"positive": 1, "negative": -1})

    alexa = alexa.rename(columns={"verified_reviews": "text", "feedback": "label"})
    alexa["label"] = alexa["label"].map({1: 1, 0: -1})

    # Combine all
    combined = pd.concat([imdb, alexa], ignore_index=True)
    combined.dropna(subset=["text", "label"], inplace=True)
    return combined

# Clean text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Balance dataset
def balance_data(df):
    pos = df[df["label"] == 1]
    neg = df[df["label"] == -1]
    min_size = min(len(pos), len(neg))

    pos_balanced = resample(pos, replace=False, n_samples=min_size, random_state=42)
    neg_balanced = resample(neg, replace=False, n_samples=min_size, random_state=42)

    balanced_df = pd.concat([pos_balanced, neg_balanced])
    return balanced_df.sample(frac=1, random_state=42)  # shuffle

# Train and evaluate models
def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC()
    }
    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model

    return best_model

# Main logic
def main():
    df = load_data()
    df["text"] = df["text"].apply(preprocess)
    df = balance_data(df)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Save best model and vectorizer
    joblib.dump(best_model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Training complete. Best model saved.")

if __name__ == "__main__":
    main()

