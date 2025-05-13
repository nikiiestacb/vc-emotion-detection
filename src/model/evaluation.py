import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load model and test data
try:
    clf = pickle.load(open('model.pkl', 'rb'))
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Model file 'model.pkl' not found!")
    exit(1)

# Load test data
test_data = pd.read_csv('./data/features/test_bow.csv')

# Ensure no missing values
test_data = test_data.fillna(0)  # Replace NaNs with 0 or another strategy

# Separate features and labels
X_test = test_data.iloc[:, :-1].values  # All columns except the last one
y_test = test_data.iloc[:, -1].values  # The last column as the label

# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Assumes binary classification

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# AUC: Handle multi-class (if applicable) with 'macro' averaging, otherwise binary
try:
    auc = roc_auc_score(y_test, y_pred_proba)
except ValueError:  # In case of multi-class classification, handle the exception
    auc = None
    print("⚠️ AUC score could not be calculated for multi-class classification.")

# Save metrics
metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'AUC': auc
}

# Write metrics to a JSON file
metrics_filename = 'metrics.json'  # Correct the file name
with open(metrics_filename, 'w') as file:
    json.dump(metrics_dict, file, indent=4)

print(f"✅ Metrics saved to {metrics_filename}")
