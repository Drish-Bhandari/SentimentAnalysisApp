# Sentiment Analysis of Amazon Alexa Product Reviews

This project classifies Amazon Alexa product reviews into Positive, Neutral, or Negative using machine learning models.

## 🔧 Features
- Upload CSV file with reviews
- Predict sentiment for each review
- Flask web app with styled UI

## 🧠 Technologies Used
- Python, Flask
- Scikit-learn (Naive Bayes, SVM, Logistic Regression)
- Pandas, HTML/CSS

## 📊 Output
Displays predicted sentiment + downloadable result.

## 📂 File Structure
- `app.py`: Flask app code
- `models/`: Pickled ML model
- `templates/`: Web interface
- `static/`: Stylesheet
- `sample_reviews.csv`: Example data

## 🖥️ Run Locally
```bash
pip install -r requirements.txt
python app.py
