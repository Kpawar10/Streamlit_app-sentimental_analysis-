import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import io

# --- Page config ---
st.set_page_config(
    page_title="Senlysis | Sentiment Analysis",
    page_icon="🧠",
    layout="wide"
)

# --- Download NLTK data ---
@st.cache_resource
def load_nltk():
    nltk.download('stopwords', quiet=True)
    return stopwords.words('english')

stop_words = load_nltk()
port_stem = PorterStemmer()

# --- Load model and vectorizer ---
import gdown
import os

@st.cache_resource
def load_model():
    if not os.path.exists('sentiment_model.pkl'):
        gdown.download(
            'https://drive.google.com/file/d/1zeBHOK4Pow7jJrnkBMtuONs0usDICjKI/view?usp=drive_link',
            'sentiment_model.pkl', quiet=False
        )
    if not os.path.exists('vectorizer.pkl'):
        gdown.download(
            'https://drive.google.com/file/d/1V0fq8wvqrmq28BGx_JslqrPmoFtZgDga/view?usp=drive_link',
            'vectorizer.pkl', quiet=False
        )
    if not os.path.exists('label_encoder.pkl'):
        gdown.download(
            'https://drive.google.com/file/d/1RMuk6syQxXrBb6Pq_pvrGLsTueeho0tZ/view?usp=drive_link',
            'label_encoder.pkl', quiet=False
        )
    model = pickle.load(open('sentiment_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, vectorizer, le

# --- Preprocessing ---
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [port_stem.stem(w) for w in text if w not in stop_words]
    return ' '.join(text)

# --- Predict ---
def predict_sentiment(text):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    label = label_map[pred]
    confidence = round(max(proba) * 100, 2)
    return label, confidence, proba

# --- Emoji & color map ---
def get_style(label):
    return {
        'Positive': ('😊', '🟢', '#28a745'),
        'Neutral':  ('😐', '🟡', '#ffc107'),
        'Negative': ('😠', '🔴', '#dc3545'),
    }[label]

# --- Session state for history ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 Senlysis")
    st.markdown("**Sentiment Analysis Tool**")
    st.markdown("Trained on 3 million Instagram Google Play reviews.")
    st.divider()
    st.markdown("**Model Details**")
    st.markdown("- Algorithm: Logistic Regression")
    st.markdown("- Features: TF-IDF Bigrams (20K)")
    st.markdown("- Classes: Positive / Neutral / Negative")
    st.markdown("- Test Accuracy: ~60%")
    st.divider()
    st.markdown("**Dataset**")
    st.markdown("[Kaggle: 3M Instagram Reviews](https://www.kaggle.com/datasets/bwandowando/3-million-instagram-google-store-reviews)")
    st.divider()
    if st.button("🗑️ Clear History"):
        st.session_state.history = []

# --- Main ---
st.title("🧠 Senlysis — Sentiment Analysis")
st.markdown("Analyze the sentiment of any text or upload a CSV to analyze in bulk.")

tab1, tab2 = st.tabs(["✍️ Single Text", "📂 Bulk CSV Analysis"])

# =====================
# TAB 1: Single Text
# =====================
with tab1:
    user_input = st.text_area(
        "Enter a sentence or review:",
        height=150,
        placeholder="e.g. The app crashes every time I open it!"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            label, confidence, proba = predict_sentiment(user_input)
            emoji, dot, color = get_style(label)

            st.divider()
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"### Result: {emoji} **{label}**")
                st.markdown(f"Confidence: **{confidence}%**")

                # Confidence bar
                st.progress(int(confidence))

            with col_b:
                st.markdown("**Class Probabilities**")
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Neutral', 'Positive'],
                    'Probability': [round(p * 100, 1) for p in proba]
                })
                st.bar_chart(prob_df.set_index('Sentiment'))

            # Add to history
            st.session_state.history.append({
                'Text': user_input[:80] + ('...' if len(user_input) > 80 else ''),
                'Sentiment': label,
                'Confidence': f"{confidence}%"
            })

    # Analysis History
    if st.session_state.history:
        st.divider()
        st.markdown("#### 📋 Analysis History (this session)")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

# =====================
# TAB 2: Bulk CSV
# =====================
with tab2:
    st.markdown("Upload a CSV with a column of text/reviews to analyze all rows at once.")
    st.markdown("**Required:** CSV must have a column named `text` or `review`.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"Loaded **{len(df)} rows**. Preview:")
        st.dataframe(df.head(3), use_container_width=True)

        # Auto-detect text column
        text_col = None
        for col in df.columns:
            if col.lower() in ['text', 'review', 'review_text', 'comment', 'content']:
                text_col = col
                break

        if not text_col:
            text_col = st.selectbox("Select the text column:", df.columns)
        else:
            st.success(f"Auto-detected text column: **{text_col}**")

        if st.button("🚀 Analyze All Rows", type="primary"):
            with st.spinner("Analyzing..."):
                results = []
                progress = st.progress(0)
                total = len(df)

                for i, row in enumerate(df[text_col].fillna('').astype(str)):
                    label, confidence, _ = predict_sentiment(row)
                    results.append({'sentiment': label, 'confidence': confidence})
                    progress.progress((i + 1) / total)

                df['sentiment'] = [r['sentiment'] for r in results]
                df['confidence_%'] = [r['confidence'] for r in results]

            st.success("Done!")

            # Summary
            st.markdown("#### Summary")
            col1, col2, col3 = st.columns(3)
            counts = df['sentiment'].value_counts()
            col1.metric("😊 Positive", counts.get('Positive', 0))
            col2.metric("😐 Neutral", counts.get('Neutral', 0))
            col3.metric("😠 Negative", counts.get('Negative', 0))

            st.bar_chart(counts)

            # Full results
            st.markdown("#### Results")
            st.dataframe(df, use_container_width=True)

            # Download
            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv_out,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )