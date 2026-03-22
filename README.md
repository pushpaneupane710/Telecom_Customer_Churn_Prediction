# **Telecom Customer Churn Prediction**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-444876?style=for-the-badge&logo=seaborn&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Joblib](https://img.shields.io/badge/Joblib-00A98F?style=for-the-badge) ![Deployed on Streamlit](https://img.shields.io/badge/Deployed-Streamlit-FF4B4B?style=for-the-badge) 

**Telecom Customer Churn Prediction App live at:**​  [![Live App](https://img.shields.io/badge/Live%20App-Click%20Here-green?style=for-the-badge)](https://telecomcustomerchurnprediction-f8ntnyptytjxpbxsdlb8qu.streamlit.app/) 

An end-to-end **Data Analytics + Machine Learning project** that analyzes telecom customer behavior and predicts churn using a deployed **Streamlit web application**.

## Problem Statement:

Telecom companies face significant revenue loss due to customer churn. The problem is to analyze customer data to understand the patterns and factors leading to churn, and to build a machine learning model that can accurately predict which customers are likely to leave. This enables businesses to take proactive actions to improve customer retention.

Churn: The rate at which customers stop doing business with the company, often measures as the percentage of lost customers or subscribers over a specific period.

**Task:** Why customers are churning out? What are the characteristics of those customers who are churners? How to retain them? Predict who will churn in future?

## Data Understanding:

The dataset contains **7032 customer records**.

#### Key Features:

-   `gender` – Customer gender
    
-   `SeniorCitizen` – Whether customer is senior
    
-   `tenure` – Number of months with company
    
-   `Contract` – Contract type
    
-   `PaymentMethod` – Payment type
    
-   `MonthlyCharges` – Monthly bill
    
-   `TotalCharges` – Total amount paid
    
-   `Churn` – Target variable (Yes/No)
    

---

### Project Executive Summary:

This project focuses on performing in-depth Exploratory Data Analysis (EDA) on telecom dataset to get insights. The objective is to conducting univariate and bivariate analyses to identify insights about churners and provide **actionable business recommendations** to improve retention and reduce revenue loss. Predict which customers are likely to leave a telecom service using historical data.

This helps telecom companies reduce churn, improve customer retention, and increase revenue.

The analysis follows a structured approach:

-   Implementing data cleaning and preprocessing techniques for real-world data.
    
-   Showcasing key churn drivers
    
-   Univariate analysis
    
-   Bivariate analysis
    
-   Generating insightful EDA reports highlighting key churn drivers.
    
-   A predictive model that accurately identifies customers at risk of churning.
    
-   An interactive Streamlit web application for predicting churn probability.
    
-   A deployed machine learning application.
    

---

## Business Insights

1.  Senior citizens are more likely to churn. Indicates need for segment-specific strategies
    
2.  People with no partners are more likely to churn
    
3.  Mothly contracts are more likely to churn because they are free customers
    
4.  People who pay via electronic check are more likely to churn
    
5.  Monthly charges and total charges are positively correlated
    
6.  Churn is high when monthly charges are high
    

---

## Technical Implementation

-   **Environment:** Jupyter Notebook (VS Code) for analysis and model building, Streamlit for deployment
-   **Language:** Python

### Libraries & Tools Used

-   **Data Processing:** Pandas, NumPy
-   **Data Visualization:** Matplotlib, Seaborn
-   **Machine Learning:** Scikit-learn (AdaBoost Classifier, preprocessing, evaluation)
-   **Model Persistence:** Joblib
-   **Web App Framework:** Streamlit

---

## ⚠️ Challenges & Solutions

### 🔴 Challenge 1: Data Quality Issues

-   `TotalCharges` column contained missing and non-numeric values
    
-   This caused errors during preprocessing
    

✅ **Solution:**

-   Converted column to numeric using `pd.to_numeric(errors='coerce')`
    
-   Removed missing values to ensure clean dataset
    

---

### 🔴 Challenge 2: Categorical Data Handling

-   Many features were in text format
    
-   Machine learning models require numerical input
    

✅ **Solution:**

-   Applied **One-Hot Encoding** using `pd.get_dummies()`
    
-   Used `drop_first=True` to avoid multicollinearity
    

---

### 🔴 Challenge 3: Feature Scaling

-   Features had different ranges (e.g., tenure vs charges)
    
-   This affected model performance
    

✅ **Solution:**

-   Applied **StandardScaler** to normalize feature values
    
-   Ensured consistent scaling during training and prediction
    

---

### 🔴 Challenge 4: Model Selection & Stability

-   Initially explored advanced models like XGBoost
    
-   Faced dependency and environment compatibility issues
    

✅ **Solution:**

-   Selected **AdaBoost Classifier** for stability and performance
    
-   Achieved reliable results with simpler implementation
    

---

### 🔴 Challenge 5: Environment Conflicts

-   Faced issues due to mixing **Anaconda and virtual environments**
    
-   Errors like NumPy mismatch and XGBoost failures occurred
    

✅ **Solution:**

-   Switched to **uv-based clean virtual environment**
    
-   Ensured consistent library versions across the project
    

---

### 🔴 Challenge 6: Deployment Compatibility

-   Streamlit Cloud does not fully support `pyproject.toml`
    
-   Dependency installation failed initially
    

✅ **Solution:**

-   Generated `requirements.txt` from project dependencies
    
-   Used minimal required libraries for faster deployment
    

---

### 🔴 Challenge 7: Consistency Between Training & Prediction

-   Risk of mismatch between training data preprocessing and app input

✅ **Solution:**

-   Replicated the same preprocessing pipeline in Streamlit app
    
-   Ensured identical feature engineering and scaling
    

---

## 💡 Key Takeaway

This project demonstrates the ability to:

-   Handle real-world data challenges
    
-   Debug environment and dependency issues
    
-   Build stable and deployable machine learning systems
    
-   Deliver business-ready solutions
    

---

## 👤 Connect With Me

-   📧 Email: [pushpaneupane710@gmail.com](mailto:pushpaneupane710@gmail.com)
-   💼 LinkedIn: [Pushpa Neupane](https://www.linkedin.com/in/pushpa-neupane-5759a6263/)