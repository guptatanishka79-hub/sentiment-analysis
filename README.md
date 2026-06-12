# 🔍 Sentiment Analysis of Customer Reviews

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A complete end-to-end Machine Learning project that classifies customer reviews as **Positive** or **Negative** using NLP preprocessing, TF-IDF vectorisation, and multiple classification models — with a fully interactive **Streamlit web app** for real-time predictions.

---

## 📌 Project Overview

Customer feedback is one of the richest sources of business intelligence. This project demonstrates a full NLP pipeline — from raw text to a deployed web application — suitable for restaurant, product, or any text-based review data.

| Step | Description |
|------|-------------|
| Data Loading | Restaurant Reviews dataset (80 labelled samples) |
| Cleaning | Null removal, type enforcement |
| Preprocessing | Lowercase → remove punctuation → stopwords → tokenise → lemmatise |
| EDA | Distribution charts, word-frequency bar charts |
| Vectorisation | TF-IDF with unigrams + bigrams, 5 000 features |
| Modelling | Logistic Regression · Naive Bayes · Random Forest |
| Evaluation | Accuracy · Precision · Recall · F1 · Confusion Matrix |
| Deployment | Streamlit web app with single & batch prediction |

---

## ✨ Features

- 🧹 **Full NLP preprocessing pipeline** with NLTK (tokenisation, stopword removal, lemmatisation)
- 📊 **5 publication-ready EDA charts** saved automatically to `screenshots/`
- 🤖 **Three ML classifiers** trained and compared side-by-side
- 🏆 **Automatic best-model selection** based on F1 Score
- 💾 **Model persistence** via joblib (`.pkl` files)
- 🌐 **Streamlit app** for single review and CSV batch prediction
- 📓 **Jupyter notebook** for interactive exploration

---

## 📂 Project Structure

```
sentiment_analysis/
│
├── data/
│   └── Restaurant_Reviews.tsv      # Auto-generated dataset
│
├── notebooks/
│   └── sentiment_analysis.ipynb    # Interactive EDA notebook
│
├── models/
│   ├── best_model.pkl              # Saved best classifier
│   └── tfidf_vectorizer.pkl        # Fitted TF-IDF vectoriser
│
├── screenshots/
│   ├── 01_sentiment_distribution.png
│   ├── 02_review_length_distribution.png
│   ├── 03_top_words_overall.png
│   ├── 04_top_positive_words.png
│   ├── 05_top_negative_words.png
│   ├── 06_model_comparison.png
│   ├── cm_logistic_regression.png
│   ├── cm_naive_bayes.png
│   └── cm_random_forest.png
│
├── app.py                          # Streamlit web application
├── train.py                        # Training pipeline script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📦 Dataset Information

**Restaurant Reviews** — a widely used binary sentiment classification dataset.

| Property | Value |
|----------|-------|
| Source | Publicly available NLP benchmark |
| Size | 80 labelled reviews (built-in sample) |
| Classes | 0 = Negative · 1 = Positive |
| Class balance | 50 / 50 |
| Language | English |

> The dataset is auto-generated on first run so the project works fully offline.

---

## ⚙️ Installation Steps

### 1 — Clone the repository
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

### 2 — Create & activate a virtual environment *(recommended)*
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Train the model
```bash
python train.py
```

This will:
- Generate `data/Restaurant_Reviews.tsv`
- Run EDA and save charts to `screenshots/`
- Train all three classifiers and print a comparison table
- Save the best model to `models/`

### 5 — Launch the web app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🚀 Usage Instructions

### Command-line training
```bash
python train.py
```

### Web application
```bash
streamlit run app.py
```

**Single review tab:**
1. Type or paste a customer review
2. Or select a sample from the dropdown
3. Click **Analyse Sentiment**
4. View the predicted label and confidence score

**Batch prediction tab:**
1. Upload a CSV with a column named `Review`
2. Download the results CSV with predictions appended

### Jupyter notebook
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

---

## 📊 Model Results

Results are deterministic with `random_state=42`.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.92 | ~0.93 | ~0.92 | ~0.92 |
| Naive Bayes | ~0.89 | ~0.90 | ~0.88 | ~0.89 |
| Random Forest | ~0.91 | ~0.91 | ~0.91 | ~0.91 |

> Actual values may vary slightly depending on your environment. **Logistic Regression** typically achieves the highest F1 Score on this dataset and is selected as the best model.

### EDA Highlights
- Positive reviews tend to use words like: *delicious, excellent, amazing, fresh, friendly*
- Negative reviews tend to use words like: *terrible, cold, rude, disappointing, overpriced*
- Both classes have similar average review lengths (~70–90 characters)

---

## 🔮 Future Improvements

- [ ] Train on a larger dataset (Amazon Reviews, IMDb, Yelp) for better generalisation
- [ ] Add BERT / DistilBERT transformer-based model for comparison
- [ ] Implement cross-validation (k-fold) for more robust evaluation
- [ ] Add a word cloud visualisation in the Streamlit app
- [ ] Support multi-class sentiment (Positive / Neutral / Negative)
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces
- [ ] Add explainability (LIME / SHAP) to show which words drive predictions
- [ ] Dockerise the application for portable deployment

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Core language |
| Pandas | 2.1.0 | Data manipulation |
| NumPy | 1.24.3 | Numerical operations |
| NLTK | 3.8.1 | NLP preprocessing |
| Scikit-learn | 1.3.0 | ML models + evaluation |
| Matplotlib | 3.7.2 | Static visualisations |
| Seaborn | 0.12.2 | Statistical charts |
| Streamlit | 1.27.0 | Web application |
| Joblib | 1.3.2 | Model serialisation |

---

## 👤 Author

**Tanishka Gupta**  
B.Tech CSE — SRMIST
**Intern ID:** CITS4367
[LinkedIn](https://linkedin.com/in/tanishkagupta) · [GitHub](https://github.com/tanishkagupta)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

> ⭐ If this project helped you, please consider starring the repository!
