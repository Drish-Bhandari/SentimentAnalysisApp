from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Try possible columns
        text_col = None
        for col in df.columns:
            if "review" in col.lower():
                text_col = col
                break

        if not text_col:
            return jsonify({"error": "No column with review text found. Try columns like 'review' or 'verified_reviews'."})

        df.dropna(subset=[text_col], inplace=True)
        if df.empty:
            return jsonify({"error": "No valid review text found in the file."})

        df['clean_text'] = df[text_col].astype(str).apply(preprocess)
        X = vectorizer.transform(df['clean_text'])
        df['prediction'] = model.predict(X)

        # Count results
        total = len(df)
        pos_count = len(df[df['prediction'] == 1])
        neg_count = len(df[df['prediction'] == -1])
        neu_count = len(df[df['prediction'] == 0]) if 0 in df['prediction'].values else 0

        # Percentages
        pos_percent = round((pos_count / total) * 100, 2)
        neg_percent = round((neg_count / total) * 100, 2)
        neu_percent = round((neu_count / total) * 100, 2)

        # Prepare summary
        summary = f"Out of {total} reviews, {pos_percent}% are Positive, {neg_percent}% are Negative, and {neu_percent}% are Neutral."

        # Top reviews
        top_pos = df[df['prediction'] == 1][text_col].head(5).tolist()
        top_neg = df[df['prediction'] == -1][text_col].head(5).tolist()
        top_neu = df[df['prediction'] == 0][text_col].head(5).tolist() if neu_count > 0 else []

        # CSV string
        output_csv = df[[text_col, 'prediction']].rename(columns={'prediction': 'Sentiment'})
        output_csv['Sentiment'] = output_csv['Sentiment'].map({1: 'Positive', -1: 'Negative', 0: 'Neutral'})
        csv_string = output_csv.to_csv(index=False)

        return jsonify({
            "summary": summary,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "pos_percent": pos_percent,
            "neg_percent": neg_percent,
            "neu_percent": neu_percent,
            "top_positive": top_pos,
            "top_negative": top_neg,
            "top_neutral": top_neu,
            "csv": csv_string
        })

    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)