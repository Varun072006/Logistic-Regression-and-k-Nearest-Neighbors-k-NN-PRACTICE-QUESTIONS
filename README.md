# Machine Learning Practice — Logistic Regression & k-NN

This repository contains Python implementations of several supervised learning tasks using **Logistic Regression** and **k-Nearest Neighbors (k-NN)**. The tasks include binary classification, multiclass classification, and regression with cross-validation. All datasets are assumed to be in CSV format.

## Introduction

This project demonstrates the application of two fundamental machine learning algorithms:

1. **Logistic Regression** — used for binary and multiclass classification tasks.  
2. **k-Nearest Neighbors (k-NN)** — used for classification with cross-validation and regression tasks.  

The goal is to understand the effect of feature scaling, hyperparameter tuning (for k-NN), and evaluation metrics in supervised learning. The project also shows how to select the best model parameters using cross-validation and how to interpret results using standard metrics.

## Tasks

### 1. Email Spam Classification
- **Model:** Logistic Regression (binary)
- **Features:** `word_free`, `word_offer`, `word_click`, `num_links`, `num_caps`, `sender_reputation`
- **Target:** `is_spam`
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix
- **Scaling:** StandardScaler applied to features

### 2. Customer Churn Prediction
- **Model:** Logistic Regression (binary)
- **Features:** `tenure_months`, `monthly_charges`, `support_tickets`, `is_premium`, `avg_usage_hours`
- **Target:** `churn`
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix
- **Scaling:** StandardScaler applied to features

### 3. Disease Stage Classification
- **Model:** Multiclass Logistic Regression (multinomial)
- **Features:** `age`, `b1`, `b2`, `b3`, `b4`
- **Target:** `stage` (0, 1, 2)
- **Evaluation Metrics:** Accuracy, Macro-F1, Weighted-F1, Confusion Matrix
- **Scaling:** StandardScaler applied to features

### 4. Flowers Classification (Iris Dataset)
- **Model:** k-NN Classifier
- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Target:** `species`
- **Hyperparameter Tuning:** 5-fold Cross-Validation to choose best `k` ∈ {1,3,…,25}
- **Evaluation Metrics:** Best `k`, CV Score, Test Accuracy, Confusion Matrix
- **Scaling:** StandardScaler applied to features

### 5. Airbnb Prices Prediction
- **Model:** k-NN Regressor
- **Features:** `size_m2`, `distance_center_km`, `rating`, `num_reviews`
- **Target:** `price`
- **Hyperparameter Tuning:** 5-fold Cross-Validation to choose best `k` ∈ {1,3,…,25}
- **Evaluation Metrics:** CV RMSE, Test RMSE, Test R²
- **Scaling:** StandardScaler applied to features

## Evaluation & ROC Curves

For classification tasks (Email Spam & Customer Churn), ROC curves were generated to evaluate the trade-off between true positive and false positive rates. Example images of ROC curves can be included here:

![ROC Curve Example](images/roc_email_spam.png)
![ROC Curve Example](images/roc_customer_churn.png)

*Note: Place your ROC images in an `images/` folder within the repository.*

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib (optional, for plotting ROC curves)

## Conclusion

This repository illustrates the practical usage of **Logistic Regression** and **k-NN** algorithms in real-world datasets. It demonstrates:

- Feature scaling and preprocessing
- Model evaluation using standard metrics (Accuracy, F1, ROC-AUC, R²)
- Hyperparameter tuning using cross-validation
- Interpretation of confusion matrices and ROC curves  # Logistic-Regression-and-k-Nearest-Neighbors-k-NN-PRACTICE-QUESTIONS
