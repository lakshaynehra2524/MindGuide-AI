
---

# 🧠 2. EDGE_PLAN.md (Your Roadmap / Upgrade Plan)

```markdown
# 🚀 MindGuide-AI Edge Plan

This document outlines the next-level improvements to enhance model performance, scalability, and real-world usability.

---

## 🎯 1. Model Improvements

### Current Issues
- High bias toward "neutral" predictions
- Weak text understanding

### Planned Upgrades
- Replace basic models with:
  - Random Forest
  - XGBoost
- Experiment with ensemble models
- Hyperparameter tuning (GridSearchCV)

---

## 🧠 2. NLP Enhancement

### Current State
- Basic vectorization (likely CountVectorizer)

### Improvements
- Switch to TF-IDF (with n-grams)
- Try pre-trained embeddings:
  - Word2Vec
  - GloVe
- Future goal:
  - BERT-based emotion classification

---

## ⚖️ 3. Data Improvements

### Current Problems
- Class imbalance (neutral dominant)
- Limited dataset diversity

### Solutions
- Apply SMOTE (oversampling)
- Collect more labeled data
- Add real-world journaling samples

---

## ⚙️ 4. Feature Engineering

- Normalize numerical features
- Add interaction features:
  - sleep × energy
  - mood × intensity
- Reduce noisy inputs

---

## 🧪 5. Evaluation Improvements

- Add confusion matrix
- Track precision, recall, F1-score
- Cross-validation instead of single split

---

## 🌐 6. Deployment Plan

- Streamlit Cloud deployment
- Docker containerization
- API version using FastAPI

---

## 🎨 7. UI/UX Improvements

- Better UI design
- Add charts for predictions
- History tracking for users

---

## 🔐 8. Reliability

- Input validation
- Handle empty text cases
- Add fallback predictions

---

## 🧩 Final Goal

Transform MindGuide-AI from:
> Academic ML Project → Real-world AI Product
