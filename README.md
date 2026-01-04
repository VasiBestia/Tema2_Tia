# Salary Prediction using Tree-Based Regression

This project focuses on predicting employee salaries using tree-based regression models. It was developed as part of the **TIA (Tehnici de Inteligență Artificială)** course for Assignment 2.

---

## 📌 Project Overview
The primary goal is to analyze the provided dataset and build a predictive model that estimates salaries using tree-based architectures. The project involves comprehensive data preprocessing and hyperparameter tuning using GridSearchCV.

## 📊 Dataset Description
The model utilizes the `salary.csv` dataset, which includes the following features :
* **Age**: Employee age.
* **Gender**: Male or Female.
* **Education Level**: Bachelor's, Master's, or PhD.
* **Job Title**: Professional designation .
* **Years of Experience**: Total professional experience.
* **Salary**: The target variable for regression.

## ⚙️ Methodology

### 1. Data Processing
Following laboratory practices, the data underwent the following transformations:
* **Cleaning**: Removal of null values (e.g., empty records found at index 172 and 260) and duplicate entries.
* **Ordinal Encoding**: "Education Level" was manually mapped to preserve its logical hierarchy.
* **Label Encoding**: Applied to "Gender" and "Job Title" to convert categorical text into numerical data.
* **Data Splitting**: The dataset was segmented into 80% training and 20% testing sets.

### 2. Model Training and Evaluation
Multiple models were trained using different combinations of hyperparameters:

| Model | Hyperparameters Optimized |
| :--- | :--- |
| **Decision Tree** | `max_depth`, `min_samples_leaf`, `min_samples_split` |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split` |

## 📈 Performance Results
The models achieved the following metrics during evaluation:

### **Random Forest (Best Model)**
* **R² Score**: 0.9412
* **MAE**: 8,581.05
* **RMSE**: 11,878.14
* **Best Params**: `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 50}`

### **Decision Tree**
* **R² Score**: 0.9206
* **MAE**: 9,750.00
* **RMSE**: 13,799.74
* **Best Params**: `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5}`

## 🚀 Submission Details
* **File Name**: `script.py`
* **Dataset**: Use only the provided dataset
