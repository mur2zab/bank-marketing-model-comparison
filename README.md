# Bank Marketing Campaign Prediction System

**ML Assignment 2 - Machine Learning Classification**

A comprehensive machine learning system for predicting customer subscription to bank term deposits using six different classification algorithms with an interactive web interface.

---

## Dataset Description

### Bank Marketing Dataset
**Source:** UCI Machine Learning Repository

The dataset contains direct marketing campaign data from a Portuguese banking institution. The marketing campaigns were based on phone calls, with the goal of determining if a client would subscribe to a term deposit.

**Dataset Statistics:**
- **Total Instances:** 45,211 records
- **Total Features:** 17 features
- **Target Variable:** `y` - Has the client subscribed to a term deposit? (binary: yes/no)
- **Class Distribution:** Highly imbalanced dataset
  - No subscription: ~88.7%
  - Subscription: ~11.3%

**Feature Categories:**
1. **Client Information:** age, job, marital status, education, default status
2. **Financial Information:** housing loan, personal loan
3. **Campaign Data:** contact type, month, day of week, campaign contacts
4. **Previous Campaign:** number of contacts, days since last contact, previous outcome
5. **Socio-Economic Context:** employment variation rate, consumer price index, consumer confidence index, euribor rate, number of employees

**Data Format:** CSV with semicolon (`;`) delimiter

---

## Problem Statement

Predict whether a client will subscribe to a term deposit based on their demographic information, previous campaign interactions, and economic indicators. This is a **binary classification** problem with highly imbalanced classes, where the positive class (subscription) represents only 11.3% of the data.

**Business Objective:** Enable the bank to:
- Target high-potential customers efficiently
- Optimize marketing campaign resources
- Improve conversion rates for term deposit subscriptions
- Reduce campaign costs by focusing on likely subscribers

---

## Model Performance Results

### Evaluation Metrics

All models were evaluated using six comprehensive metrics:
- **Accuracy:** Overall correctness of predictions
- **AUC (ROC-AUC):** Area under ROC curve, measures class separation ability
- **Precision:** Proportion of correct positive predictions
- **Recall:** Proportion of actual positives correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **MCC:** Matthews Correlation Coefficient for imbalanced datasets

### Results Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| **Logistic Regression** | 0.1201 | 0.5123 | 0.1174 | 1.0000 | 0.2101 | 0.0203 |
| **Decision Tree** | 0.8994 | 0.8507 | 0.5845 | 0.4839 | 0.5295 | 0.4763 |
| **K-Nearest Neighbors** | 0.8481 | 0.4977 | 0.1108 | 0.0425 | 0.0615 | -0.0042 |
| **Naive Bayes** | 0.1172 | 0.5001 | 0.1170 | 1.0000 | 0.2095 | 0.0054 |
| **Random Forest** | 0.9055 | 0.9214 | 0.6890 | 0.3497 | 0.4639 | 0.4472 |
| **XGBoost** | 0.9066 | 0.9281 | 0.6323 | 0.4811 | 0.5464 | 0.5012 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Demonstrated extremely poor performance with only 12.01% accuracy, essentially predicting the majority class almost exclusively. While achieving perfect recall (1.0000) by classifying nearly all instances as positive, the precision was abysmal at 0.1174, resulting in an F1 score of just 0.2101. The AUC of 0.5123 indicates the model barely performs better than random guessing. This suggests severe issues with default hyperparameters or class imbalance handling, making it unsuitable for this task without significant modifications. |
| **Decision Tree** | Achieved strong performance with 89.94% accuracy and an excellent AUC of 0.8507, demonstrating good class separation. The model maintained a balanced approach with precision of 0.5845 and recall of 0.4839, yielding an F1 score of 0.5295. The MCC of 0.4763 indicates solid correlation between predictions and actual values. The tree structure provides excellent interpretability, making it easy to understand decision rules, though it may be prone to overfitting without proper pruning. |
| **K-Nearest Neighbors** | Showed moderate accuracy of 84.81% but struggled significantly with minority class prediction. The extremely low recall of 0.0425 and precision of 0.1108 indicate the model failed to effectively identify positive cases. The negative MCC (-0.0042) and AUC near random (0.4977) suggest the model couldn't learn meaningful patterns from the feature space. The distance-based approach appears poorly suited for this imbalanced dataset without significant preprocessing. |
| **Naive Bayes** | Exhibited similar failure patterns to Logistic Regression with only 11.72% accuracy, classifying almost everything as the positive class. Despite perfect recall (1.0000), the precision of 0.1170 and F1 of 0.2095 indicate massive overprediction of the positive class. The AUC of 0.5001 and near-zero MCC (0.0054) show the model essentially performs at random chance level. The strong independence assumption appears violated for this dataset's feature structure. |
| **Random Forest** | Delivered excellent performance with 90.55% accuracy and the second-highest AUC of 0.9214, showcasing strong predictive capability. The model achieved the best precision among all models at 0.6890, though recall was moderate at 0.3497, indicating conservative positive predictions. The F1 score of 0.4639 and MCC of 0.4472 demonstrate good overall balance. The ensemble approach effectively reduced overfitting while maintaining interpretability through feature importance analysis. |
| **XGBoost** | Achieved the best overall performance with 90.66% accuracy and the highest AUC of 0.9281, demonstrating superior class discrimination ability. The model showed excellent balance with precision of 0.6323 and the highest recall of 0.4811, resulting in the best F1 score of 0.5464 and MCC of 0.5012. The gradient boosting framework effectively handled the class imbalance and complex feature interactions. This is the recommended model for production deployment due to its superior predictive performance and robustness. |

---

## Interactive Web Application

### Features

1. **File Upload:** Upload CSV datasets with automatic delimiter detection
2. **Sample Data:** Download sample test data for demonstration
3. **Model Selection:** Choose from 6 trained ML models via radio buttons
4. **Metrics Display:** View all 6 evaluation metrics simultaneously
5. **Confusion Matrix:** Visual heatmap with dark theme styling
6. **Download Predictions:** Export results with original data as CSV
7. **Comparison Matrix:** Compare up to 3 models side-by-side

### User Interface
- **Design:** Dark minimalist theme with IBM Plex typography
- **Layout:** Centered three-step workflow (Upload → Select → Predict)
- **Navigation:** Clean interface without sidebar
- **Responsive:** Works across desktop, tablet, and mobile

---

## How to Run Locally

```bash
# Clone repository
git clone https://github.com/mur2zab/bank-marketing-model-comparison.git
cd bank-marketing-model-comparison

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

# Access at http://localhost:8501
```

---

## Assignment Details

**Course:** Machine Learning - M.Tech (AIML)  
**Institution:** BITS Pilani  
**Assignment:** ML Assignment 2  
**Student:** Murtuza Bagasrawala  
**ID:** 2025AA05246  
**Date:** February 15, 2026

---

## Key Insights

**Best Models:**
1. **XGBoost** - 90.66% accuracy, 0.9281 AUC (Recommended)
2. **Random Forest** - 90.55% accuracy, 0.9214 AUC
3. **Decision Tree** - 89.94% accuracy, 0.8507 AUC

**Models Needing Improvement:**
- Logistic Regression and Naive Bayes require class imbalance handling
- KNN needs better feature engineering

---

## Links

- **GitHub:** https://github.com/mur2zab/bank-marketing-model-comparison
- **Live App:** https://murtuza-bank-marketing-model-comparison.streamlit.app/
- **Dataset:** https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## Author

**Murtuza Bagasrawala**  
ID: 2025AA05246  
M.Tech (AIML) - BITS Pilani

---

**Last Updated:** February 15, 2026