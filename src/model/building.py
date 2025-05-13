import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml

# Load model configuration from params.yaml
config_path = 'C:/desktop/mlops/emotion_detection/params.yaml'  # Specify the correct path to params.yaml
try:
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)['model_building']
    print("✅ Configuration loaded successfully.")
except FileNotFoundError:
    print(f"❌ File not found: {config_path}")
    exit(1)
except yaml.YAMLError as e:
    print(f"❌ Error loading YAML configuration: {e}")
    exit(1)

# Load the training data
train_data = pd.read_csv('./data/features/train_bow.csv')

# Ensure no missing values
train_data = train_data.fillna(0)  # Replace NaNs with 0 or another strategy

# Separate features and labels
X_train = train_data.iloc[:, 0:-1].values  # All columns except the last one
y_train = train_data.iloc[:, -1].values  # The last column as the label

# Print the shapes of features and labels
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Initialize the GradientBoostingClassifier with parameters from the config
clf = GradientBoostingClassifier(
    n_estimators=params['n_estimators'], 
    learning_rate=params['learning_rate']
)

# Train the model
clf.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(clf, model_file)

print(f"✅ Model saved to {model_filename}")

