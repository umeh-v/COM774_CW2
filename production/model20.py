import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import mlflow
import mlflow.sklearn

# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--trainingdata', type=str, required=True, help= 'dataset for training')
parser.add_argument('--testingdata', type=str, required=True, help= 'dataset for testing')

# Parse the arguments
args = parser.parse_args()

mlflow.autolog()

# Load the datasets
train_dataset = pd.read_csv(args.trainingdata)
test_dataset = pd.read_csv(args.testingdata)

#train_dataset = pd.read_csv(r"C:\Users\user\Documents\AI_MSc\COM774\CW2\train_dataset.csv")
#test_dataset = pd.read_csv(r"C:\Users\user\Documents\AI_MSc\COM774\CW2\test_dataset.csv")

# Encoding the target variable
label_encoder = LabelEncoder()
train_dataset['Activity'] = label_encoder.fit_transform(train_dataset['Activity'])
test_dataset['Activity'] = label_encoder.transform(test_dataset['Activity'])

train_dataset.head()

train_dataset.shape

# EDA Visualizations
plt.figure(figsize=(12, 6))
sns.countplot(x='Activity', data=train_dataset)
plt.title('Distribution of Activities')
plt.xlabel('Activity')
plt.ylabel('Frequency')
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45)
plt.show()

# Splitting the dataset
features = train_dataset.drop('Activity', axis=1)
labels = train_dataset['Activity']
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, random_state=42)

# Data Transformation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Feature Selection - Selecting the top 20 features
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances and sort them in descending order
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:50]  # Selecting top 20 indices

# Select the top 20 features from the train and validation sets
X_train_selected = X_train[:, indices]
X_val_selected = X_val[:, indices]

# Feature Selection Visualization for Top 20 Features
plt.figure(figsize=(12, 6))
plt.title("Top 20 Feature Importances")
plt.bar(range(50), importances[indices])
plt.xticks(range(50), features.columns[indices], rotation=90)
plt.show()

print(f"Number of features selected: {X_train_selected.shape[1]}")

# Model Selection and Hyperparameter Tuning
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_selected, y_train)
best_model = grid_search.best_estimator_

# Model Evaluation on Validation Set
y_val_pred = best_model.predict(X_val_selected)
print("Validation Set Evaluation:\n")
print(classification_report(y_val, y_val_pred))

# Confusion Matrix Visualization (Validation Set)
conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.title('Confusion Matrix for Validation Set')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Preparing Test Dataset with Selected Features
test_features = test_dataset.drop('Activity', axis=1)
test_features_scaled = scaler.transform(test_features)
test_features_selected = test_features_scaled[:, indices]  # Applying the same feature selection to the test set
y_test = test_dataset['Activity']

# Final Evaluation on Test Set with Selected Features
y_test_pred = best_model.predict(test_features_selected)
print("\nTest Set Evaluation:\n")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix Visualization (Test Set)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(test_conf_matrix, annot=True, fmt='g')
plt.title('Confusion Matrix for Test Set')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
