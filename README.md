# Salary Prediction using Tree-Based Regression

[cite_start]This project focuses on predicting employee salaries using tree-based regression models[cite: 1]. [cite_start]It was developed as part of the **TIA (Tehnici de Inteligență Artificială)** course for Assignment 2[cite: 1].

---

## 📌 Project Overview
[cite_start]The primary goal is to analyze the provided dataset and build a predictive model that estimates salaries using tree-based architectures[cite: 3, 5]. [cite_start]The project involves comprehensive data preprocessing and hyperparameter tuning using GridSearchCV[cite: 5].

## 📊 Dataset Description
[cite_start]The model utilizes the `salary.csv` dataset [cite: 9][cite_start], which includes the following features[cite: 11]:
* [cite_start]**Age**: Employee age[cite: 11].
* [cite_start]**Gender**: Male or Female[cite: 11].
* [cite_start]**Education Level**: Bachelor's, Master's, or PhD[cite: 11].
* [cite_start]**Job Title**: Professional designation[cite: 11].
* [cite_start]**Years of Experience**: Total professional experience[cite: 11].
* [cite_start]**Salary**: The target variable for regression[cite: 11].

## ⚙️ Methodology

### 1. Data Processing
[cite_start]Following laboratory practices[cite: 3], the data underwent the following transformations:
* [cite_start]**Cleaning**: Removal of null values (e.g., empty records found at index 172 and 260) and duplicate entries[cite: 13, 15].
* [cite_start]**Ordinal Encoding**: "Education Level" was manually mapped to preserve its logical hierarchy[cite: 11].
* **Label Encoding**: Applied to "Gender" and "Job Title" to convert categorical text into numerical data.
* **Data Splitting**: The dataset was segmented into 80% training and 20% testing sets.

### 2. Model Training and Evaluation
[cite_start]Multiple models were trained using different combinations of hyperparameters[cite: 5]:

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
* [cite_start]**File Name**: `Nume_Prenume_Grupa.py` or `.ipynb`[cite: 8].
* [cite_start]**Dataset**: Use only the provided dataset[cite: 9].
* [cite_start]**Deadline**: 06.01.2026, 23:59[cite: 10].
