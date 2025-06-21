# Predicting-loan-default

## 📌 Overview

The objective of this project is to develop a supervised machine learning model to **predict loan default** based on a variety of observable and legally usable client and transaction characteristics. This is a critical task in the banking sector, which seeks to minimise risk and ensure financial stability by accurately forecasting which loans are most likely to default.

We utilise the **Berka Dataset**, a comprehensive, anonymised financial dataset from a Czech bank, which includes information on clients, accounts, transactions, and loans.

---

## 📂 Data Description

The dataset comprises multiple relational files capturing different facets of banking activity:

| File Name         | Observations | Description |
|------------------|--------------|-------------|
| `account.asc`     | 4,500        | Static characteristics of bank accounts |
| `client.asc`      | 5,369        | Demographic characteristics of clients |
| `disp.asc`        | 5,369        | Mapping between clients and accounts |
| `order.asc`       | 6,471        | Information on credit card payment orders |
| `transaction.asc` | 1,056,320    | Detailed account transactions |
| `loan.asc`        | 682          | Records of loans granted to accounts |
| `card.asc`        | 892          | Details of credit cards issued |
| `district.asc`    | 77           | Demographic statistics of districts |

⚠️ **Note:** Given the richness of the dataset, we will selectively use only the most relevant tables for this prediction task.

---

## 🔄 Workflow Summary

1. **Load and Explore the Data**  
   Load selected datasets and perform initial exploratory analysis to understand distributions, missing data, and relationships.

2. **Preprocessing & Feature Engineering**  
   Merge datasets (e.g., `client`, `account`, `loan`, `transaction`), handle missing values, and create features relevant to loan risk (e.g., average balance, transaction count, district unemployment rate).

3. **Model Training**  
   Apply classification models such as Logistic Regression, Random Forest, and Gradient Boosting. Tune hyperparameters using cross-validation.

4. **Evaluation**  
   Use accuracy, precision, recall, F1-score, and ROC-AUC to evaluate model performance on a hold-out validation set.

---

## 🧠 Model Objective

**Target Variable:** `loan default (yes/no)`  
**Goal:** Predict whether a loan will default based on financial and demographic characteristics prior to or at loan issuance.

---

## 🛠️ Tools & Libraries

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost (optional)
- Jupyter Notebook

---


