#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # força uso de CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ---------------------------
# 1. Load Dataset
# ---------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# ---------------------------
# 2. Visualizations - Exploratory
# ---------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=data, palette='viridis')
plt.title('Distribution of Outcome')
plt.xticks([0, 1], ['No Disease', 'Disease'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

data.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle('Feature Distributions')
plt.tight_layout()
plt.show()

# ---------------------------
# 3. Data Preprocessing
# ---------------------------
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Aplicar antes da separação em X e y
data_clean = remove_outliers(data, ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'Age'])


X = data.drop('Outcome', axis=1)
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------
# 4. Build Neural Network
# ---------------------------
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ---------------------------
# 5. Train Model
# ---------------------------
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), verbose=0)

# ---------------------------
# 6. Evaluate Model
# ---------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")

y_probs = model.predict(X_test).ravel()
y_pred = (y_probs > 0.5).astype("int32")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 7. Confusion Matrix (Normalized)
# ---------------------------
cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
disp.plot(cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.show()

# ---------------------------
# 8. Training History: Accuracy and Loss
# ---------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------
# 9. ROC Curve and AUC
# ---------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()

