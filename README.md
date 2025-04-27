# 🛍️ NLP Project: Automated Customer Reviews

This project uses Natural Language Processing (NLP) and Transformer-based models to automate the analysis of Amazon customer reviews. It includes review classification, category clustering, and review summarization using state-of-the-art language models.

---

## 🚀 Project Overview

### 🎯 Goals
- Automatically classify customer sentiment (positive, neutral, negative)
- Cluster products into meaningful categories
- Generate recommendation-style summaries using generative AI

---

## 📁 Dataset

Primary Dataset:  
[📦 Amazon Product Reviews on Kaggle](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data?select=Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv)

---

## 📊 Task Breakdown

### ✅ Task 0: Preprocessing
Mapped star ratings to sentiment classes:
- 1–2 ⭐ → Negative
- 3 ⭐ → Neutral
- 4–5 ⭐ → Positive

---

### 🧠 Task 1: Sentiment Classification

Used Transformer models to classify reviews into sentiment classes.

🔗 Pretrained and fine-tuned model:  
[Download Classification Model](https://drive.google.com/file/d/1y8_ss47dlFzLCql3hXdAZ_XfnK-hfUsl/view?usp=sharing)

**Models tried:**
- `bert-base-uncased`


📈 Evaluation:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix

---

### 🧱 Task 2: Product Category Clustering

Clustered product titles using **TF-IDF** and **KMeans** into 4 main meta-categories:
- Kindle e-readers
- Kids tablets
- Echo devices
- Regular tablets

**Keywords from clusters** revealed meaningful groupings (e.g., `"echo", "alexa", "speaker"` for Echo devices).

---

### 📝 Task 3: Review Summarization

Used **T5** for summarizing review content into recommendation blog-style posts per category.  
Each summary includes:
- Top 3 recommended products
- Key differences between them
- Top complaints
- Worst product in the group

---

## 🖥️ Deployment

The project is deployed as a web app using Streamlit.

👉 [Launch the Review Sentiment Classifier App](nlpautomated-customer-reviews-ufqqsoqp4drqcvg6g5hs5o)

---

## 🔧 Tech Stack

- Python, Pandas, Scikit-learn
- Transformers (Hugging Face)
- Sentence-BERT, BERT, RoBERTa, T5
- Streamlit (for deployment)
- Matplotlib/Seaborn (for visualization)

---


Built with ❤️ for applied NLP and practical.
