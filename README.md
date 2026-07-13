# 🕵️ Sherlock

Sherlock is a Machine Learning data preprocessing and model recommendation platform that analyzes raw datasets, recommends preprocessing steps, prepares data for Machine Learning, trains multiple ML models, evaluates their performance, and predicts outcomes on unseen data.

Built completely from scratch while learning Machine Learning fundamentals.

---

# ✨ Features

## 📂 Dataset Investigation

- Upload CSV datasets
- Dataset preview
- Dataset overview (Rows, Columns, Missing Values, Duplicates)
- Column-wise summary
- Automatic problem type detection
  - Binary Classification
  - Multi-class Classification
  - Regression

---

## 🔎 Data Quality Analysis

### Missing Value Detection

- Detects missing values in every column
- Suggests intelligent imputation:
  - Mean
  - Median
  - Mode
  - Forward Fill
  - Backward Fill

### Duplicate Detection

- Detects duplicate rows
- Displays duplicate records
- Recommends removal

### Constant Column Detection

- Detects columns having only one unique value
- Recommends removal

### Outlier Detection

- Detects outliers using the IQR method
- Shows:
  - Lower Bound
  - Upper Bound
  - Number of Outliers
  - Preview of Outlier Values
- Recommends capping using IQR bounds

---

## 🧠 Smart Preprocessing Recommendations

### Encoding Suggestions

Automatically recommends:

- Label Encoding
- One-Hot Encoding
- High Cardinality Detection

### Scaling Suggestions

Automatically selects:

- StandardScaler
- RobustScaler
- MinMaxScaler

based on data distribution and outliers.

---

## ⚙️ Automated Data Cleaning Pipeline

One-click preprocessing that performs:

- Missing Value Handling
- Duplicate Removal
- Constant Column Removal
- Outlier Capping
- Feature Encoding
- Feature Scaling

with a live processing tracker.

---

## 🤖 Machine Learning

Automatically recommends models based on the detected problem type.

### Classification Models

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost

### Regression Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)
- XGBoost Regressor

---

## 📈 Model Evaluation

Displays model performance using:

### Classification

- Accuracy
- Precision
- Recall
- F1 Score

### Regression

- RMSE
- MAE
- R² Score

---

## 🏆 Model Leaderboard

Keeps track of every trained model.

Displays:

- Model Name
- Accuracy / R²
- Precision
- Recall
- F1 Score

Automatically highlights the current best-performing model.

---

## 📊 Feature Importance

For tree-based models Sherlock displays:

- Ranked Feature Importance Table
- Feature Importance Bar Chart

---

## 🔮 Prediction Engine

Upload a new dataset to:

- Predict outcomes using the trained model
- View prediction results
- Download predictions as CSV

---

# 🛠️ Tech Stack

### Languages

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit

---

# 📚 Machine Learning Concepts Covered

Sherlock implements and demonstrates:

- Data Cleaning
- Missing Value Imputation
- Duplicate Handling
- Constant Feature Removal
- Outlier Detection (IQR)
- Feature Encoding
- Feature Scaling
- Problem Type Detection
- Train-Test Split
- Model Selection
- Classification
- Regression
- Feature Importance
- Model Evaluation Metrics
- Model Comparison
- Prediction on Unseen Data

---

# 🎯 Goal

Sherlock isn't just another AutoML tool.

It is being built as a learning-first project to understand every stage of the Machine Learning pipeline—from raw data inspection to model deployment—while implementing each concept from scratch instead of relying on black-box automation.

---

Every feature is built incrementally while learning Machine Learning.
