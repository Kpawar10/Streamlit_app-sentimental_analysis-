# Sentiment Analysis on Instagram App Reviews

This project analyzes 3 million Instagram reviews from the Google Play Store to predict sentiment using machine learning. It handles multilingual text, preprocesses it, balances the dataset, and trains a logistic regression model for classification.

## 📂 Dataset
- Source: Kaggle
- [Link](https://www.kaggle.com/datasets/bwandowando/3-million-instagram-google-store-reviews)
- Note: Due to size constraints, a small sample is included in `/data/`.

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn, NLTK
- TfidfVectorizer
- Logistic Regression
- Pickle (model serialization)

## 📊 Data Preprocessing
- Cleaned missing data
- Labeled reviews into: `positive`, `neutral`, `negative`
- Used stemming and removed stopwords
- Converted text into TF-IDF features (bigrams)

## ⚙️ Model
- Trained a `Logistic Regression` model
- Achieved:
  - Training Accuracy: `61%`
  - Test Accuracy: `60%`
  - Baseline Accuracy: ~33%

## 📁 Folder Structure
- `src/`: Scripts for preprocessing, model training, prediction
- `models/`: Saved `.sav` model file
- `data/`: Sample CSV or link to full dataset
- `notebook/`: Optional Jupyter Notebook version

## ▶️ How to Run
```bash
pip install -r requirements.txt
python src/sentiment_analysis.py
