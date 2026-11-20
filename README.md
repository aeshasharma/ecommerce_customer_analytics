# 🛒 E-Commerce Customer Analytics & Recommendation System

A machine-learning-powered e-commerce analytics platform that analyzes customer purchasing behavior, segments customers, recommends products, predicts churn risk, and estimates customer monetary value.

🚀 **Live Demo:** https://ecom-customer-insights.streamlit.app

## 📌 Project Overview

This project uses machine learning techniques to transform historical e-commerce transaction data into actionable business insights.

It includes four major components:

| Problem | Algorithm | Output |
|---|---|---|
| Customer Segmentation | RFM + KMeans | VIP, Regular, New, At Risk, Lost |
| Product Recommendation | Implicit ALS | Personalized Product Recommendations |
| Churn Prediction | XGBoost Classifier | Customer Churn Risk |
| Customer Value Prediction | XGBoost Regressor | Predicted Monetary Value |

## 📊 Dataset

**Dataset:** UCI Online Retail II

The dataset contains transaction records from a UK-based online retailer covering December 2009 to December 2011.

Dataset features:
- Customer IDs
- Product codes and descriptions
- Purchase quantities
- Unit prices
- Transaction dates

## ⚙️ Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Customer Segmentation
5. Product Recommendation
6. Churn Prediction
7. Customer Value Prediction
8. Interactive Dashboard Development

## 1. Customer Segmentation

**Algorithm:** KMeans Clustering

Customer segmentation is performed using RFM analysis:

- Recency: Days since the last purchase.
- Frequency: Number of purchases.
- Monetary: Total customer spending.

Five customer segments are identified:

- VIP Customers
- Regular Customers
- New Customers
- At Risk Customers
- Lost Customers

**Silhouette Score:** 0.369

## 2. Product Recommendation System

**Algorithm:** Alternating Least Squares (ALS)

The recommendation engine uses implicit collaborative filtering to generate personalized product recommendations.

Features:
- Customer-product interaction matrix.
- Purchase quantity as implicit feedback.
- Personalized product ranking.
- Previously purchased products filtering.
- Popularity-based fallback for cold-start customers.
- TF-IDF-based product similarity.

**Evaluation:**

| Metric | Score |
|---|---:|
| Precision@10 | 0.1875 |
| NDCG@10 | 0.1829 |

## 3. Customer Churn Prediction

**Algorithm:** XGBoost Classifier

The model identifies customers who may stop purchasing.

Project-specific churn definition:
No purchase for 180 or more days.

**Evaluation:**

| Metric | Score |
|---|---:|
| ROC-AUC | 0.932 |
| F1-Score | 0.81–0.88 |

## 4. Customer Value Prediction

**Algorithm:** XGBoost Regressor

This model estimates historical customer monetary value using customer behavioral features.

**Evaluation:**

| Metric | Score |
|---|---:|
| R² Score | 0.969 |
| MAE | £157.13 |

Note: This implementation predicts historical customer spending rather than future lifetime value.

## 📈 Streamlit Dashboard

The project provides an interactive dashboard with four main sections:

**Home**
- Overall customer insights.
- Business performance metrics.

**Segmentation**
- Customer segment visualization.
- Recency, Frequency and Monetary analysis.

**Recommendations**
- Select a customer.
- Generate personalized product recommendations.
- View ranked product suggestions.

**Churn & Customer Value**
- Customer churn information.
- Predicted customer monetary value.
- Customer behavioral insights.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Machine Learning | Scikit-learn |
| Clustering | KMeans |
| Recommendation | Implicit ALS |
| Content Similarity | TF-IDF |
| Churn Prediction | XGBoost |
| Regression | XGBoost |
| Model Persistence | Joblib, Pickle |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |

## 📂 Project Structure

```text
ecommerce_customer_analytics/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── churn_model.pkl
│   ├── clv_model.pkl
│   ├── als_model.pkl
│   └── als_artifacts.pkl
│
└── data/
    ├── rfm_customers.csv
    └── product_catalog.csv
```

*The structure above represents the main project components; adjust it to match the actual repository.*

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aeshasharma/ecommerce_customer_analytics.git
```

### 2. Navigate to the Project

```bash
cd ecommerce_customer_analytics
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser through the local Streamlit server.

## 📊 Model Evaluation Summary

| Component | Algorithm | Metric | Result |
|---|---|---|---:|
| Segmentation | KMeans | Silhouette Score | 0.369 |
| Recommendation | ALS | Precision@10 | 0.1875 |
| Recommendation | ALS | NDCG@10 | 0.1829 |
| Churn Prediction | XGBoost | ROC-AUC | 0.932 |
| Churn Prediction | XGBoost | F1-Score | 0.81–0.88 |
| Customer Value | XGBoost | R² | 0.969 |
| Customer Value | XGBoost | MAE | £157.13 |

## 🎯 Key Learning Outcomes

- Practical implementation of machine learning algorithms.
- Customer behavior analysis using RFM techniques.
- Collaborative filtering and recommendation systems.
- Classification and regression model development.
- Data preprocessing and feature engineering.
- Model evaluation and performance analysis.
- Interactive dashboard development using Streamlit.
- Deployment of machine learning applications.

## 👩‍💻 Author

**Aesha Sharma**

MCA | Brainware University

GitHub: https://github.com/aeshasharma

## ⭐ Conclusion

This project combines customer segmentation, personalized product recommendations, churn prediction, and customer value estimation into a single machine-learning-powered e-commerce analytics platform.

It helps transform transaction data into meaningful customer insights and supports data-driven business decisions.



