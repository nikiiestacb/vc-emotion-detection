import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import yaml

# Directly specify the path to params.yaml
config_path = 'C:/desktop/mlops/emotion_detection/params.yaml'

# Load the configuration from params.yaml
try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print("✅ Configuration loaded successfully.")
except FileNotFoundError:
    print(f"❌ File not found: {config_path}")
    exit(1)
except yaml.YAMLError as e:
    print(f"❌ Error loading YAML configuration: {e}")
    exit(1)

# Extract the max_features from the configuration file
max_features = config['feature_engineering']['max_features']

# Load the processed train and test datasets
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

# Fill missing values with empty string in both train and test data
train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# Separate features and labels
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=max_features)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

# Convert the transformed train and test data into DataFrames
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# Store the feature-engineered data in the features directory
data_path = os.path.join("data", "features")
os.makedirs(data_path, exist_ok=True)

# Save the processed train and test data to CSV files
train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)

print("✅ Feature engineering completed and files saved.")
