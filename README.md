# 🏦 Loan Approval System Using Bagging Random Forest

## 📌 Project Overview

This project predicts whether a loan application will be **Approved** or **Rejected** using Machine Learning. The model is trained on historical loan application data and learns patterns from applicant information such as income, education, employment status, loan amount, and financial assets.

The project follows a complete machine learning workflow including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and model serialization for deployment.

---

## 🎯 Objectives

- Predict loan approval status accurately.
- Perform data preprocessing and feature encoding.
- Scale numerical features for better model performance.
- Train a Bagging ensemble model using Random Forest.
- Evaluate the model using classification metrics.
- Save the trained model for future deployment.

---

## 📂 Dataset

**Dataset:** `loan_approval_dataset.csv`

### Features

- Loan ID
- Number of Dependents
- Education
- Self Employed
- Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value

### Target Variable

- **Loan Status**
  - Approved
  - Rejected

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Removed extra spaces from column names.
- Checked dataset information.
- Checked missing values.
- Checked duplicate records.
- Encoded categorical features using **LabelEncoder**.
- Encoded the target variable (`loan_status`).
- Split the dataset into training and testing sets.
- Applied **StandardScaler** for feature scaling.

---

## 📊 Exploratory Data Analysis (EDA)

The project includes visualization of:

- Loan Status Distribution
- Confusion Matrix

These visualizations help understand class distribution and evaluate model performance.

---

## 🤖 Machine Learning Model

The model used in this project is:

### Bagging Classifier

**Base Estimator**

- Random Forest Classifier

### Why Bagging?

Bagging (Bootstrap Aggregating) improves prediction performance by combining multiple Random Forest models trained on different subsets of the data. This reduces variance and increases model stability.

---

## 📈 Model Evaluation

The trained model was evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix

These metrics provide insight into the model's prediction performance on unseen data.

---

## 💾 Model Saved

The trained model is saved using Joblib.

```python
bagging_rf_model.pkl
```

This model can be loaded later for deployment in a web application.

## 📁 Project Structure

```
Loan_Approval_Prediction/
│
├── Loan_Approval_Prediction.ipynb
├── loan_approval_dataset.csv
├── bagging_rf_model.pkl
├── README.md
└── requirements.txt
```

## 🚀 Workflow

```
Dataset
   │
   ▼
Load Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Label Encoding
   │
   ▼
Feature Scaling
   │
   ▼
Train-Test Split
   │
   ▼
Bagging Classifier
(Random Forest Base Estimator)
   │
   ▼
Model Evaluation
   │
   ▼
Model Saving
```

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/zaheerbalochtech-gif/Loan_Approval_System.git
```

Move into the project directory

```bash
cd Loan_Approval_Prediction
```

Install the required libraries

```bash
pip install -r requirements.txt
```

Launch google colab Notebook

```bash
google colab notebook
```

Open

```
Loan_Approval_Prediction.ipynb
```


## 📷 Results

### Loan Status Distribution

<p align="center">
  <img src="https://github.com/user-attachments/assets/90cf02c2-cd49-420c-915a-f4b29af2abe2" width="700">
</p>

---

### Confusion Matrix

<p align="center">
  <img src="https://github.com/user-attachments/assets/a9154501-e4ab-4a39-a9cf-71b41bd2d7bb" width="700">
</p>

---

### Classification Report

<p align="center">
  <img src="https://github.com/user-attachments/assets/6cbee417-9aac-4810-a844-f996d8d967cb" width="700">
</p>


##Model Accuracy
Accuracy Score: 0.9723207948899929

---

## 🔮 Future Improvements

- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Compare Bagging with XGBoost, AdaBoost, and Gradient Boosting.
- Deploy the trained model using Streamlit.
- Add feature importance visualization.
- Save preprocessing objects (LabelEncoder and StandardScaler) for deployment.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Data preprocessing
- Feature encoding
- Feature scaling
- Ensemble learning
- Random Forest
- Bagging Classifier
- Model evaluation
- Classification metrics
- Confusion matrix visualization
- Machine Learning workflow

---

## 👨‍💻 Author

**Zaheer Ahmed**

BS Computer Science Student  
University of Turbat

- GitHub: https://github.com/zaheerbalochtech-gif
- LinkedIn: https://linkedin.com/in/www.linkedin.com/in/zaheer-ahmed-54300338b

---

## ⭐ If you found this project helpful, please consider giving it a star.
