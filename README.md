# 🧠 Senlysis — Sentiment Analysis App

A deployed machine learning web application that classifies Instagram app reviews as **Positive**, **Neutral**, or **Negative** using NLP and Logistic Regression.

🔗 **Live Demo:** [senlysis.streamlit.app](https://senlysis.streamlit.app)

---

## 📸 Features

- **Single text analysis** — paste any review and get instant sentiment + confidence score
- **Bulk CSV upload** — analyze thousands of rows at once and download results
- **Confidence breakdown** — see probability scores for all 3 classes as a bar chart
- **Session history** — tracks all analyses in the current session

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression |
| Features | TF-IDF Bigrams (max 20,000 features) |
| Training Accuracy | 61% |
| Test Accuracy | 60% |
| Baseline (random) | ~33% |
| Classes | Positive / Neutral / Negative |

> The model performs at ~2x above random chance on a balanced 3-class problem. Future improvements: fine-tuned DistilBERT (see roadmap below).

---

## 🗂️ Dataset

- **Source:** [3 Million Instagram Google Play Reviews (Kaggle)](https://www.kaggle.com/datasets/bwandowando/3-million-instagram-google-store-reviews)
- **Size:** ~3 million multilingual reviews
- **Preprocessing:** dropped nulls, removed non-ASCII, stemming, stopword removal
- **Class balancing:** upsampling minority classes to match positive class count

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Scikit-learn** — TF-IDF vectorizer, Logistic Regression
- **NLTK** — stemming, stopwords
- **Streamlit** — web app deployment
- **Pandas / NumPy** — data processing
- **Pickle** — model serialization

---

## 📁 Project Structure

```
├── app.py                      # Streamlit web app
├── sentimental_analysis_.py    # Training script
├── sentimental_analysis__.ipynb # Jupyter notebook (EDA + training)
├── requirements.txt            # Dependencies
└── README.md
```

> **Note:** Model files (`sentiment_model.pkl`, `vectorizer.pkl`) are not included due to size (~26MB). Download from [Google Drive](#) or retrain using the notebook.

---

## ▶️ Run Locally

```bash
git clone https://github.com/Kpawar10/Streamlit_app-sentimental_analysis-
cd Streamlit_app-sentimental_analysis-
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Roadmap / Future Improvements

- [ ] Replace TF-IDF + LR with fine-tuned **DistilBERT** for higher accuracy
- [ ] Add **SHAP explainability** — show which words drove the prediction
- [ ] Support **multilingual reviews** (the dataset contains 40+ languages)
- [ ] Add **emoji and slang** handling in preprocessing
- [ ] Compare multiple models side-by-side in the UI

---

## 👤 Author

**Kritika Pawar** — [GitHub](https://github.com/Kpawar10)