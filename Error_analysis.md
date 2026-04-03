This document analyzes the key issues observed in the model predictions and identifies possible root causes.

---

## 🚨 Problem Statement

The model frequently predicts:
- "neutral" emotional state
- Similar outputs for different inputs

---

## 🔍 Observed Issues

### 1. Prediction Bias

- Model heavily favors "neutral"
- Low variation across inputs

### 2. Low Sensitivity to Text Input

- Journal text has minimal impact
- Predictions driven mostly by categorical features

---

## 🧠 Root Cause Analysis

### 1. Class Imbalance

- Training data likely contains more "neutral" samples
- Model learns majority class dominance

---

### 2. Weak Text Representation

- Vectorizer may have:
  - Small vocabulary
  - Poor training corpus
- Leads to sparse or zero-heavy vectors

---

### 3. Feature Scaling Issues

- Numerical features not normalized
- Larger values dominate model decisions

---

### 4. Label Encoding Limitations

- LabelEncoder introduces ordinal relationships
- Model may interpret categories incorrectly

---

### 5. Model Underfitting

- Model too simple to capture patterns
- Fails to differentiate subtle emotional signals

---

## 📊 Diagnostic Evidence

- Probability outputs skewed toward one class
- Minimal change in predictions despite varied inputs

---

## 🛠️ Fixes Implemented / Planned

- Apply class balancing techniques
- Upgrade to TF-IDF vectorization
- Introduce feature scaling
- Try advanced models (RandomForest, XGBoost)
- Improve dataset diversity

---

## 🧪 Future Evaluation Plan

- Use confusion matrix
- Track F1-score per class
- Perform cross-validation

---

## ✅ Conclusion

The issue is not with the UI or deployment, but with:
- Data quality
- Feature representation
- Model selection

Improving these will significantly enhance prediction diversity and accuracy.
