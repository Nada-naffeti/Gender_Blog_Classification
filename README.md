# Gender Classification from Blog Text

**Predicting author gender from writing style using NLP and Deep Learning**  
TF-IDF · Traditional ML · DistilBERT · BiLSTM · Blog Authorship Corpus

---

## Overview

This project builds a complete machine learning pipeline to classify the gender of blog authors based on their writing style and word usage. Three families of approaches are trained and compared: traditional ML models, a transformer (DistilBERT), and a recurrent neural network (BiLSTM).

| | |
|---|---|
| **Dataset** | [Blog Authorship Corpus — Kaggle](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus) |
| **Task** | Binary classification (male / female) |
| **Input** | Raw blog post text |
| **Sample size** | 10,000 documents |

---

## Methodology

### 1. Data Loading & Exploration
- Gender distribution analysis (bar chart, pie chart)
- Text length distribution by gender
- Average token count per gender

### 2. Text Preprocessing
- Lowercasing, Unicode normalization, URL and HTML removal
- Tokenization, stopword removal, lemmatization (NLTK)
- Vocabulary filtering: words appearing in fewer than 5 documents are discarded

### 3. Feature Engineering : Custom TF-IDF
- Manual implementation of TF-IDF from scratch
- L2 normalization of feature vectors

### 4. Vocabulary Analysis
- Most frequent words per gender
- Distinctive words algorithm: words 1.5x more frequent in one gender are flagged as gender-specific

### 5. Models Trained

**Traditional ML (on TF-IDF features)**
- Logistic Regression
- Linear SVC
- Multinomial Naive Bayes
- Random Forest

**Transformer : DistilBERT**
- Fine-tuned on blog text with PyTorch
- AdamW optimizer, cross-entropy loss

**Recurrent Neural Network : BiLSTM**
- Embedding layer + Bidirectional LSTM
- Dropout regularization (0.5 / 0.3)
- Trained with TensorFlow/Keras

### 6. Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrix per model
- Final comparison table and visualization across all models

---

## Results

| Model | Type | Accuracy |
|-------|------|----------|
| **DistilBERT** | Transformer | **82.12%** |
| Multinomial NB | Traditional ML | 80.24% |
| Logistic Regression | Traditional ML | 79.20% |
| Linear SVC | Traditional ML | 78.88% |
| Random Forest | Traditional ML | 77.96% |
| BiLSTM | Deep Learning | 75.65% |

> DistilBERT achieves the best accuracy (82.12%), outperforming all traditional ML models. Multinomial Naive Bayes is the strongest traditional baseline at 80.24% with much lower training cost.

---

## Tech Stack

```
scikit-learn · NLTK · numpy · pandas · matplotlib · seaborn
PyTorch · Hugging Face Transformers (DistilBERT)
TensorFlow / Keras (BiLSTM)
```

---

## Project Structure

```
gender-blog-classification/
├── README.md
├── final-version.ipynb
└── saved_models/
    ├── distilbert_model/
    ├── bilstm_model.h5
    ├── preprocessing_components.pkl
    └── model_comparison_results.csv
```

---

## Getting Started

```bash
git clone https://github.com/Nada-naffeti/gender-blog-classification.git
cd gender-blog-classification
pip install numpy pandas matplotlib seaborn scikit-learn nltk torch transformers tensorflow
jupyter notebook final-version.ipynb
```

> The dataset is available on [Kaggle](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus). Place it at `data/blogtext.csv` before running.

---

## Author

**Nada Naffeti** — Data Science & AI Engineering Student @ [ESSAI](https://www.essai.tn/)

[LinkedIn](https://linkedin.com/in/nada-naffeti) · [GitHub](https://github.com/Nada-naffeti)
