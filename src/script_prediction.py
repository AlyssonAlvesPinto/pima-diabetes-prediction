import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.utils import class_weight

from keras_tuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import tensorflow as tf
from xgboost import XGBClassifier

# 1. Load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# 2. Replace invalid zeros with NaN and fill with median
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    data[col].replace(0, np.nan, inplace=True)
    data[col].fillna(data[col].median(), inplace=True)

# 3. Remove outliers using IQR
def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

data = remove_outliers(data, ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'Age'])

# 4. Prepare data
X = data.drop('Outcome', axis=1)
y = data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Hyperparameter optimization
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(hp.Int('units1', 32, 128, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout1', 0.0, 0.5, step=0.1)))
    model.add(Dense(hp.Int('units2', 16, 64, step=16), activation='relu'))
    model.add(Dropout(hp.Float('dropout2', 0.0, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='pima_nn'
)

tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=0)
best_model = tuner.get_best_models(num_models=1)[0]

# 6. Evaluate Neural Network
y_probs_nn = best_model.predict(X_test).ravel()
y_pred_nn = (y_probs_nn > 0.5).astype(int)
print("\n🔍 Neural Network Report:")
print(classification_report(y_test, y_pred_nn))
print(f"AUC NN: {roc_auc_score(y_test, y_probs_nn):.3f}")

# 7. Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_probs_xgb = xgb.predict_proba(X_test)[:, 1]
print("\n🌲 XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))
print(f"AUC XGB: {roc_auc_score(y_test, y_probs_xgb):.3f}")

# 8. Visualizations
# ROC
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_probs_nn)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_probs_xgb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_nn, tpr_nn, label=f'Neural Net (AUC={roc_auc_score(y_test, y_probs_nn):.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_score(y_test, y_probs_xgb):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_nn, display_labels=['No Disease', 'Disease'], cmap='Blues')
plt.title('Neural Net - Confusion Matrix')
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_xgb, display_labels=['No Disease', 'Disease'], cmap='Greens')
plt.title('XGBoost - Confusion Matrix')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Feature distributions
data.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle('Feature Distributions (After Cleaning)')
plt.tight_layout()
plt.show()

# Class distribution
sns.countplot(x='Outcome', data=data, palette='viridis')
plt.title('Target Class Distribution')
plt.show()
