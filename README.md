# 🧠 Pima Diabetes Prediction

This repository demonstrates an end-to-end machine learning pipeline for binary classification using the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). It applies both a tuned neural network and XGBoost classifier, along with data preprocessing, exploratory data analysis, outlier handling, and visualization.

---

## 🔍 Problem Statement

Diabetes is a chronic condition with significant public health impact. The goal is to accurately classify whether a patient has diabetes based on medical measurements.

---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Observations**: 768 patients
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (target variable: 1 for diabetic, 0 for non-diabetic)

---

## 💡 Key Features of the Project

✔️ Exploratory data analysis with distribution plots and correlation matrix  
✔️ Outlier detection using the IQR method  
✔️ Missing/impossible values imputed with medians  
✔️ Data standardization  
✔️ Deep Learning (Neural Network with `keras-tuner` for optimization)  
✔️ Tree-Based Model (XGBoost) for benchmarking  
✔️ ROC/AUC comparisons and confusion matrix visualizations  
✔️ Fully modular and reproducible codebase  

---

## 🧪 Models Implemented

| Model          | Library         | Notes                                   |
|----------------|------------------|------------------------------------------|
| Neural Network | TensorFlow/Keras | Tuned using `keras-tuner` (RandomSearch) |
| XGBoost        | XGBoost          | Strong tree-based benchmark              |

---


## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pima-diabetes-prediction.git
cd pima-diabetes-prediction
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## 📊 Model Evaluation Results



### 🔗 Correlation Matrix
![image alt](https://github.com/AlyssonAlvesPinto/pima-diabetes-prediction/blob/main/data/CM.png)




### 📈 Feature Distributions
![image alt](https://github.com/AlyssonAlvesPinto/pima-diabetes-prediction/blob/main/data/Distrib.png)



### 📉 Accuracy and Loss Over Epochs
![image alt](https://github.com/AlyssonAlvesPinto/pima-diabetes-prediction/blob/main/data/Train_Val.png)


### 📉 ROC Curve Comparison
![image alt](https://github.com/AlyssonAlvesPinto/pima-diabetes-prediction/blob/main/data/ROC_Comparison.png)

