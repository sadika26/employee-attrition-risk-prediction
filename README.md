# Employee Attrition Risk Prediction

## Project Overview

This project presents a Machine Learning-Based Employee Attrition Prediction and Risk Scoring System. It analyzes employee data to identify factors associated with employee attrition and estimates the probability of an employee leaving the organization.

## Objectives

- Predict employee attrition using machine learning.
- Generate an individual employee attrition risk score.
- Identify important factors influencing attrition.
- Analyze attrition risk across departments and job roles.
- Provide interactive what-if risk exploration.
- Present insights through a Streamlit dashboard.

## Dataset

The project uses the `Palo Alto Networks.csv` dataset containing 1,470 employee records and 31 original features.

The dataset includes employee information such as:

- Age
- Department
- Job Role
- Monthly Income
- Job Satisfaction
- Environment Satisfaction
- Work-Life Balance
- OverTime
- Years at Company
- Years in Current Role
- Years Since Last Promotion
- Attrition

## Data Preprocessing

The project includes:

- Categorical feature encoding
- Numerical feature scaling
- Stratified train-test split
- Class imbalance handling using class weights

## Feature Engineering

Additional features were created to improve the analysis:

- Income Per Working Year
- Promotion Delay Ratio
- Promotion Delay Flag
- Engagement Score
- Workload Stress Flag

## Machine Learning

The project evaluates machine learning models for employee attrition prediction.

The dashboard uses a Logistic Regression pipeline with class weighting to handle class imbalance.

Model evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Streamlit Dashboard

The interactive dashboard provides:

- Overall employee statistics
- Individual employee risk prediction
- Department-wise attrition analysis
- Overall risk distribution
- Adjustable risk thresholds
- Department and risk category filters
- Job-role-wise risk analysis
- Feature importance
- What-if risk exploration

## Risk Scoring

The model generates an attrition probability between 0 and 1, which is converted into a percentage-based risk score.

The dashboard provides Low, Medium, and High Risk categories using adjustable risk thresholds.

## Project Files

- `app.py` – Streamlit dashboard
- `analysis.py` – Data analysis and machine learning experiments
- `train_pipeline.py` – Dashboard model training pipeline
- `Palo Alto Networks.csv` – Dataset
- `requirements.txt` – Required Python packages
- `.pkl` files – Saved model and preprocessing objects
- `.png` files – Generated analysis visualizations

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Project Outcome

The project provides an interactive machine learning solution for analyzing employee attrition and identifying employees who may have higher attrition risk. The dashboard can be used to explore employee-level risk, organizational patterns, and factors associated with attrition.
