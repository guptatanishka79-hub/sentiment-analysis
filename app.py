"""
app.py
------
Streamlit web application for the Sentiment Analysis project.

Features:
  - Single review prediction with confidence score
  - Batch prediction via CSV upload
  - Live example reviews for quick testing
  - Model performance metrics display

Run with:
    streamlit run app.py
"""

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ─── NLTK setup ──────────────────────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "best_model.pkl")
TFIDF_PATH  = os.path.join("models", "tfidf_vectorizer.pkl")


# ─── Preprocessing (must match train.py exactly) ──────────────────────────────
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ─── Load model & vectorizer ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf


def predict_sentiment(review: str, model, tfidf):
    """Return (label, confidence_score, cleaned_text)."""
    cleaned  = preprocess_text(review)
    features = tfidf.transform([cleaned])
    pred     = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]
    confidence = float(np.max(proba))
    label = "Positive 😊" if pred == 1 else "Negative 😞"
    return label, confidence, cleaned


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    # Page config
    st.set_page_config(
        page_title="Sentiment Analyser",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ─────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #666;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        .result-box {
            padding: 1.2rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 1rem;
        }
        .positive { background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
        .negative { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
        .confidence-text { font-size: 0.95rem; color: #555; margin-top: 0.4rem; }
        .stTextArea textarea { font-size: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="main-header">🔍 Sentiment Analyser</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Machine Learning — enter a customer review to predict its sentiment.</p>', unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Dataset:** Restaurant Reviews  
        **Models trained:**
        - Logistic Regression
        - Naive Bayes
        - Random Forest

        **Best model** is selected automatically based on F1 Score.

        **Preprocessing pipeline:**
        1. Lowercasing
        2. Remove punctuation / URLs
        3. Stopword removal
        4. Tokenisation
        5. Lemmatisation

        **Features:** TF-IDF (unigrams + bigrams, 5 000 features)
        """)

        st.header("📊 Typical Model Results")
        results = {
            "Model": ["Logistic Regression", "Naive Bayes", "Random Forest"],
            "Accuracy": ["~92%", "~89%", "~91%"],
            "F1 Score": ["~0.92", "~0.89", "~0.91"],
        }
        st.dataframe(pd.DataFrame(results), hide_index=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    model, tfidf = load_model()
    if model is None:
        st.error(
            "⚠️  Model not found. Please run `python train.py` first to train and save the model.",
            icon="🚨"
        )
        st.stop()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["💬 Single Review", "📄 Batch Prediction"])

    # ── Tab 1: Single prediction ───────────────────────────────────────────────
    with tab1:
        st.subheader("Enter a Customer Review")

        # Quick-fill examples
        examples = {
            "✅ Positive example": "The food was absolutely delicious and the service was outstanding. Highly recommend!",
            "❌ Negative example": "Terrible experience. The food was cold and the waiter was very rude.",
            "🤔 Mixed example":    "The ambiance was nice but the food was a bit disappointing.",
        }
        chosen = st.selectbox("Or pick a sample review:", ["— type your own —"] + list(examples.keys()))

        default_text = examples.get(chosen, "")
        review_input = st.text_area(
            "Review text:",
            value=default_text,
            height=130,
            placeholder="e.g. 'The pasta was perfectly cooked and the staff were very friendly …'",
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyse_btn = st.button("🔍 Analyse Sentiment", use_container_width=True, type="primary")

        if analyse_btn:
            if not review_input.strip():
                st.warning("Please enter a review before clicking Analyse.")
            else:
                with st.spinner("Analysing …"):
                    label, confidence, cleaned = predict_sentiment(review_input, model, tfidf)

                css_class = "positive" if "Positive" in label else "negative"
                st.markdown(
                    f'<div class="result-box {css_class}">'
                    f'Predicted Sentiment: {label}<br>'
                    f'<span class="confidence-text">Confidence: {confidence * 100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Confidence progress bar
                st.markdown("**Confidence Score**")
                bar_colour = "normal" if confidence > 0.7 else "off"
                st.progress(confidence)

                with st.expander("🔬 View preprocessed text"):
                    st.code(cleaned if cleaned else "(empty after preprocessing)", language="text")

    # ── Tab 2: Batch prediction ────────────────────────────────────────────────
    with tab2:
        st.subheader("Upload a CSV for Batch Prediction")
        st.markdown("The CSV must have a column named **`Review`**.")

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
                if "Review" not in batch_df.columns:
                    st.error("CSV must contain a column named 'Review'.")
                else:
                    with st.spinner(f"Processing {len(batch_df)} reviews …"):
                        labels, scores = [], []
                        for rev in batch_df["Review"].fillna(""):
                            lbl, conf, _ = predict_sentiment(str(rev), model, tfidf)
                            labels.append(lbl)
                            scores.append(round(conf * 100, 1))

                    batch_df["Predicted Sentiment"] = labels
                    batch_df["Confidence (%)"]      = scores

                    st.success(f"Done! Processed {len(batch_df)} reviews.")
                    st.dataframe(batch_df, use_container_width=True)

                    csv_out = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️  Download Results",
                        data=csv_out,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<small>Built with Python · Scikit-learn · Streamlit | "
        "Sentiment Analysis Portfolio Project</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
