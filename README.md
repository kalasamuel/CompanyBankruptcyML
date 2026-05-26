"# Company Bankruptcy ML"
Company Bankruptcy Prediction: Machine Learning Project

A machine learning project that predicts company bankruptcy using financial ratio data from the Polish Companies Bankruptcy Dataset from the UCI Machine Learning Repository.

The project focuses on:
Binary classification (bankrupt vs healthy companies)
Multi-class classification (predicting bankruptcy within 1–5 years)
Explainable AI using SHAP and LIME
Handling highly imbalanced financial datasets
End-to-end preprocessing and model evaluation pipelines

Project Overview:
Bankruptcy prediction is an important task in financial risk management because it helps identify companies that may fail in the future.
This project uses five years of financial data from Polish companies to answer two major questions:
Whether a Company will go bankrupt?
If bankruptcy happens, after how many years will it occur?
The project also includes explainability techniques so that predictions made by the models can be interpreted by analysts and researchers.

Dataset Information
Dataset Source: Polish Companies Bankruptcy Dataset
Dataset Characteristics:
Total companies: 43,405
Features: 64 financial ratio attributes
File format: ARFF
Dataset split into 5 yearly files
Highly imbalanced dataset
Approximately 95% healthy companies
Approximately 5% bankrupt companies

Target Variables:
Binary Classification Target
0 -> Healthy company
1 -> Bankrupt company
Multi-Class Classification Target
0 -> No bankruptcy
1–5 -> Bankruptcy after 1 to 5 years

Important Financial Features
The dataset contains financial ratios derived from company balance sheets and income statements.

Examples include:
Attr1 → Net profit / Total assets
Attr2 → Total liabilities / Total assets
Attr3 → Working capital / Total assets
Attr6 → Retained earnings / Total assets
Attr7 → EBIT / Total assets
Attr27 → Profit on operating activities / Financial expenses
Attr29 → Logarithm of total assets

Feature definitions are stored in: Data_dictionary.txt

Notebook Workflow
1. Data Loading
Notebook:loaddata.ipynb

Tasks performed:
Loads all ARFF files,
Merges yearly datasets,
Creates target labels,
Saves dataset as poland_df.csv

2. Exploratory Data Analysis (EDA)
Notebook:EDA.ipynb
Tasks performed:
Class imbalance visualization,
Correlation analysis,
Outlier detection,
Feature distribution analysis

4. Binary Classification
Notebook:bankruptcy_prediction_binaryclassification.ipynb

Goal: Predict whether a company is bankrupt or healthy.

Models used:
Logistic Regression,
Random Forest,
XGBoost

Key preprocessing steps:
Missing value handling,
Median imputation,
Feature scaling,
IQR Winsorization,
Feature selection,
SMOTE oversampling

Evaluation metrics:
ROC-AUC
F1 Score

4. Multi-Class Classification
Notebook: bankruptcy_prediction_multi_clasiification.ipynb

Goal: Predict the number of years before bankruptcy.

Models used:
Decision Tree,
Random Forest,
Gradient Boosting,
XGBoost

Techniques used:
RandomOverSampler,
GridSearchCV,
Stratified Cross Validation,
Scikit-learn Pipelines

Evaluation metrics:
Accuracy,
Weighted F1 Score,
ROC-AUC One-vs-Rest,

5. Explainable AI
Notebook: ExplainableAI.ipynb

Explainability tools:
SHAP:
Used for:
Global feature importance,
Beeswarm plots,
Heatmaps,
Waterfall plots

LIME:
Used for:
Local prediction explanations,
Bankrupt company analysis,
Healthy company analysis


Because the dataset is heavily imbalanced, different balancing strategies are used.
Binary Classification
Technique used:
SMOTE (Synthetic Minority Oversampling Technique)

Multi-Class Classification
Technique used:
RandomOverSampler,
Oversampling is applied only to training data to prevent data leakage.

Machine Learning Models:
The project experiments with several machine learning algorithms:
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
XGBoost

XGBoost achieved the best overall performance in both classification tasks.
