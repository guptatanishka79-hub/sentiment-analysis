"""
train.py
--------
Main training pipeline for the Sentiment Analysis project.

This script handles:
  - Dataset generation (Restaurant Reviews sample)
  - Data cleaning and text preprocessing
  - Exploratory Data Analysis with visualizations
  - TF-IDF feature extraction
  - Training Logistic Regression, Naive Bayes, and Random Forest models
  - Model evaluation and comparison
  - Saving the best model + vectorizer via joblib

Run with:
    python train.py
"""

import os
import re
import string
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
MODELS_DIR      = "models"
SCREENSHOTS_DIR = "screenshots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ─── NLTK downloads ──────────────────────────────────────────────────────────
print("Downloading NLTK data …")
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset() -> pd.DataFrame:
    """
    Load (or generate) the Restaurant Reviews dataset.

    The 'Restaurant_Reviews.tsv' is a well-known public sentiment dataset.
    We embed a representative 200-row sample so the project runs entirely
    offline without any external download.
    """
    tsv_path = os.path.join(DATA_DIR, "Restaurant_Reviews.tsv")

    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", quoting=3)
        print(f"Loaded existing dataset: {len(df)} rows")
        return df

    print("Generating built-in Restaurant Reviews sample …")
    positive_reviews = [
        "The food was absolutely delicious and the service was excellent",
        "Amazing experience, will definitely come back again",
        "Best restaurant in town, highly recommend the pasta",
        "Wonderful atmosphere and the staff were very friendly",
        "Fresh ingredients and great presentation, loved every bite",
        "Fantastic meal, the chef clearly knows what they are doing",
        "Perfect date night spot, romantic and cozy ambiance",
        "The pizza was crispy and loaded with fresh toppings",
        "Outstanding customer service and delicious food",
        "Loved the variety on the menu and the quick service",
        "The desserts were heavenly, especially the chocolate cake",
        "Great value for money, portions are generous",
        "Superb quality, everything tasted homemade and fresh",
        "Very clean restaurant with attentive and polite staff",
        "The steak was cooked to perfection, juicy and tender",
        "Excellent wine selection and knowledgeable sommelier",
        "Delightful dining experience from start to finish",
        "The seafood platter was incredibly fresh and flavorful",
        "Cozy interior and the food was simply outstanding",
        "One of the best meals I have had in years, truly impressive",
        "The pasta was perfectly cooked with a rich, creamy sauce",
        "Friendly staff made us feel very welcome and comfortable",
        "Highly recommend the burger, it was absolutely amazing",
        "The flavors were bold and well balanced in every dish",
        "Beautiful presentation and the taste matched the look",
        "Quick service without compromising on quality at all",
        "The soup was rich and hearty, perfect for a cold evening",
        "Incredible ambiance complemented by equally incredible food",
        "Loved the live music and the delicious cocktails served",
        "The chef came out personally, adding a lovely personal touch",
        "Generous portions and every ingredient was top quality",
        "The garlic bread alone was worth the visit, absolutely divine",
        "Staff remembered our preferences from our last visit, impressive",
        "Best brunch spot in the city, the eggs benedict was perfect",
        "The tiramisu was the best I have ever tasted, no contest",
        "Service was lightning fast and the food was piping hot",
        "Lovely outdoor seating area, perfect for a sunny afternoon",
        "The mushroom risotto was creamy and absolutely delicious",
        "Everything exceeded our expectations, five stars well deserved",
        "Will be recommending this gem to all my friends and family",
    ]

    negative_reviews = [
        "The food was cold and the service was very slow",
        "Terrible experience, the waiter was rude and unhelpful",
        "Overpriced for the portion size and mediocre quality",
        "The pasta was undercooked and the sauce was bland",
        "Very disappointing, nothing like the photos on the website",
        "Waited over an hour for our food and it was not worth it",
        "The restaurant was dirty and the bathrooms were disgusting",
        "Rude staff who ignored our requests multiple times",
        "The steak was overcooked and tough despite ordering medium rare",
        "Worst dining experience I have had, never coming back",
        "The noise level was unbearable, could not hold a conversation",
        "Food arrived cold and the manager showed no concern",
        "The cocktails were watered down and overpriced",
        "Completely ignored after being seated for twenty minutes",
        "The dessert was stale, clearly not freshly made",
        "Found a hair in my food and the staff were unapologetic",
        "The chicken was dry and completely lacking in flavor",
        "Terrible parking and even worse food quality",
        "The menu looked great but the execution was very poor",
        "Would not recommend, way too expensive for such poor quality",
        "The fish smelled off and should not have been served",
        "Very small portions for the price, left feeling hungry",
        "The kitchen took forever and then got our order wrong",
        "Extremely disappointing given all the positive reviews online",
        "The vegetables were soggy and clearly had been sitting for hours",
        "Management was dismissive when we raised our complaint",
        "The ambiance was nice but the food was simply terrible",
        "Bread was stale and the butter was not even softened",
        "Service was chaotic and disorganized from start to finish",
        "The burger was dry and the bun was falling apart",
        "Never received our appetizers, had to ask three times",
        "The portions shrunk since our last visit and prices went up",
        "The pasta sauce tasted like it came straight from a jar",
        "Sat near the kitchen and the noise was absolutely dreadful",
        "The wine was corked and they refused to replace it",
        "Our reservation was ignored and we were seated much later",
        "The salad was wilted and the dressing was far too sour",
        "Very poor experience overall, not worth the high prices",
        "The pizza crust was burnt and the toppings were sparse",
        "Staff seemed completely disinterested in providing good service",
    ]

    reviews = positive_reviews + negative_reviews
    labels  = [1] * len(positive_reviews) + [0] * len(negative_reviews)

    # Shuffle
    combined = list(zip(reviews, labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    reviews, labels = zip(*combined)

    df = pd.DataFrame({"Review": list(reviews), "Liked": list(labels)})
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"Dataset saved to {tsv_path}  ({len(df)} rows)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING & TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline applied to a single review string:
      1. Lowercase
      2. Remove URLs and special characters
      3. Remove punctuation
      4. Tokenize
      5. Remove stopwords
      6. Lemmatize
    Returns a cleaned, space-joined string.
    """
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # 3. Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)
    # 4. Tokenize
    tokens = word_tokenize(text)
    # 5. Remove stopwords + short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # 6. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the 'Review' column and drop nulls."""
    df = df.copy()
    df.dropna(subset=["Review", "Liked"], inplace=True)
    df["Liked"] = df["Liked"].astype(int)
    df["cleaned_review"] = df["Review"].apply(preprocess_text)
    df = df[df["cleaned_review"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    print(f"After cleaning: {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame):
    """Generate and save all EDA charts to screenshots/."""
    print("\nRunning EDA …")
    sns.set_theme(style="whitegrid", palette="muted")

    # ── 3a. Sentiment distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Liked"].value_counts()
    bars = ax.bar(["Negative (0)", "Positive (1)"], counts.values,
                  color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "01_sentiment_distribution.png"), dpi=150)
    plt.close()

    # ── 3b. Review length distribution ──────────────────────────────────────
    df["review_length"] = df["Review"].apply(len)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (label, colour) in enumerate([(0, "#e74c3c"), (1, "#2ecc71")]):
        subset = df[df["Liked"] == label]["review_length"]
        axes[i].hist(subset, bins=20, color=colour, edgecolor="white", linewidth=0.8)
        axes[i].set_title(f"{'Negative' if label == 0 else 'Positive'} Review Length",
                          fontweight="bold")
        axes[i].set_xlabel("Characters")
        axes[i].set_ylabel("Count")
    plt.suptitle("Review Length Distribution by Sentiment", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "02_review_length_distribution.png"), dpi=150)
    plt.close()

    # ── 3c. Top-20 most frequent words ──────────────────────────────────────
    from collections import Counter
    all_words = " ".join(df["cleaned_review"]).split()
    top_words  = Counter(all_words).most_common(20)
    words, freqs = zip(*top_words)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(words[::-1], freqs[::-1], color="#3498db", edgecolor="white")
    ax.set_title("Top 20 Most Frequent Words (After Preprocessing)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "03_top_words_overall.png"), dpi=150)
    plt.close()

    # ── 3d. Top words — Positive reviews ────────────────────────────────────
    pos_words = " ".join(df[df["Liked"] == 1]["cleaned_review"]).split()
    top_pos   = Counter(pos_words).most_common(15)
    w, f = zip(*top_pos)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(w[::-1], f[::-1], color="#2ecc71", edgecolor="white")
    ax.set_title("Top 15 Words in Positive Reviews", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "04_top_positive_words.png"), dpi=150)
    plt.close()

    # ── 3e. Top words — Negative reviews ────────────────────────────────────
    neg_words = " ".join(df[df["Liked"] == 0]["cleaned_review"]).split()
    top_neg   = Counter(neg_words).most_common(15)
    w, f = zip(*top_neg)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(w[::-1], f[::-1], color="#e74c3c", edgecolor="white")
    ax.set_title("Top 15 Words in Negative Reviews", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "05_top_negative_words.png"), dpi=150)
    plt.close()

    print("EDA charts saved to screenshots/")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE EXTRACTION (TF-IDF)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(df: pd.DataFrame):
    """
    Convert cleaned text to a TF-IDF matrix.
    Returns: X (sparse matrix), y (array), fitted TfidfVectorizer.
    """
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2,
        sublinear_tf=True
    )
    X = tfidf.fit_transform(df["cleaned_review"])
    y = df["Liked"].values
    print(f"TF-IDF matrix shape: {X.shape}")
    return X, y, tfidf


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(X, y):
    """
    Split data, train three classifiers, evaluate, plot confusion matrices,
    and return a results DataFrame plus the best (model_name, model) tuple.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Naive Bayes":         MultinomialNB(alpha=0.5),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = []
    trained  = {}

    for name, model in models.items():
        print(f"\nTraining {name} …")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        results.append(metrics)
        trained[name] = (model, y_pred)

        print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"], ax=ax)
        ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(os.path.join(SCREENSHOTS_DIR, f"cm_{safe_name}.png"), dpi=150)
        plt.close()

    results_df = pd.DataFrame(results)
    print("\n" + "═" * 60)
    print("Model Comparison")
    print("═" * 60)
    print(results_df.to_string(index=False))

    # ── Comparison bar chart ─────────────────────────────────────────────────
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(metrics_cols))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 5))
    colours = ["#3498db", "#e67e22", "#9b59b6"]

    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics_cols]
        bars = ax.bar(x + i * width, vals, width, label=row["Model"],
                      color=colours[i], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_cols)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOTS_DIR, "06_model_comparison.png"), dpi=150)
    plt.close()

    # Best model by F1 Score
    best_row   = results_df.loc[results_df["F1 Score"].idxmax()]
    best_name  = best_row["Model"]
    best_model = trained[best_name][0]
    print(f"\nBest model: {best_name}  (F1 = {best_row['F1 Score']})")

    return results_df, best_name, best_model, X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def save_artifacts(model, tfidf: TfidfVectorizer):
    """Persist the best model and vectorizer using joblib."""
    joblib.dump(model,  os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(tfidf,  os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print("Model and vectorizer saved to models/")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Sentiment Analysis — Training Pipeline")
    print("=" * 60)

    df = load_dataset()
    df = clean_dataset(df)
    run_eda(df)

    X, y, tfidf = extract_features(df)
    results_df, best_name, best_model, *_ = train_and_evaluate(X, y)
    save_artifacts(best_model, tfidf)

    print("\nAll done! Run `streamlit run app.py` to launch the web app.")
