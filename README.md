# Spam Detection — SMS Classification with NLP

A machine learning project that classifies SMS messages as **spam** or **ham** (legitimate).
Built from scratch using Multinomial Naive Bayes, with full feature engineering, grid search, k-fold cross validation, and explainability analysis.

---

## Video Presentation

[![Watch the presentation](https://img.shields.io/badge/YouTube-Watch%20Presentation-red?logo=youtube)](https://youtu.be/wLTlNBk9ojs)

---

## Results

| Metric | Score |
|--------|-------|
| F1-score (spam class) | **0.9602** |
| PR AUC | **0.9828** |
| ROC AUC | **0.9944** |

Best configuration: `tfidf_plus_ratios` features + Naive Bayes `alpha=0.1` + SMOTE balancing

---

## Project Structure

| File | Description |
|------|-------------|
| `spam_detection.ipynb` | Main notebook — full implementation with outputs, visualizations, and explanations across all 6 parts. No need to run it; all results are pre-rendered. |
| `sms_spam.csv` | Dataset — 5,572 SMS messages labeled as spam or ham. |
| `presentation.pdf` | Project presentation slides (Hebrew). |

---

## Dataset

[SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification)

5,572 SMS messages, ~87% ham / ~13% spam (binary classification).

---

## What's Inside the Notebook

### Part 1 — Introduction & Data Loading
- Student details and AI tools usage log
- Problem description: binary classification of SMS messages
- Dataset loading, column standardization, 80/20 train/test split
- Preview of first 5 rows from each split

### Part 2 — Feature Engineering
- **Text cleaning:** lowercase, punctuation removal, whitespace normalization
- **TF-IDF vectorization** (`max_features=3000`)
- **Additional numeric features:** message length, word count, digit ratio, uppercase ratio, currency/phone presence flags
- Feature engineering shown on 2–3 examples from both train and test sets

### Part 3 — Learning Algorithm (from scratch)
- **Multinomial Naive Bayes** implemented manually (no sklearn classifier)
- Hyperparameter: `alpha` (Laplace smoothing)
- Custom `train()` and `predict()` functions
- F1-score used as primary metric (spam class only) due to class imbalance

### Part 4 — Final Training Flow
- Model trained on full training set using best configuration found in Part 6
- Examples of 2–3 messages going through the feature engineering pipeline

### Part 5 — Evaluation on Test Set
- First 5 predictions shown
- F1-score, confusion matrix, precision/recall breakdown

### Part 6 — Extensions & Experiments

| Sub-part | Description |
|----------|-------------|
| **6a** | K-fold cross validation (5-fold) + grid search infrastructure |
| **6b** | Feature engineering experiments: TF-IDF only, TF-IDF + numeric, bigrams, ratio features |
| **6c** | Hyperparameter tuning: `alpha` ∈ [0.1, 0.5, 1.0, 2.0] × `max_features` ∈ [1000, 3000, 5000] |
| **6d** | Data imbalance handling: under-sampling, over-sampling, SMOTE |
| **6e** | Additional quality metrics: Precision-Recall curve, ROC curve, AUC scores |
| **6f** | Explainability: top spam/ham words, per-message explanation, SHAP values |
| **6g** | All experiments unified in one results table; best configuration identified |

---

## Students

Niv S. · Liora R. · Idan M.
