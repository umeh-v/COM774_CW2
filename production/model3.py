# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import argparse
import mlflow
import mlflow.sklearn

# Enable automatic logging to MLflow
mlflow.autolog()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--trainingdata', type=str, required=True, help='Dataset for training')
parser.add_argument('--testingdata', type=str, required=True, help='Dataset for testing')
args = parser.parse_args()

# Load datasets
train_dataset = pd.read_csv(args.trainingdata)
test_dataset = pd.read_csv(args.testingdata)

# Encoding the target variable
label_encoder = LabelEncoder()
train_dataset['Activity'] = label_encoder.fit_transform(train_dataset['Activity'])
test_dataset['Activity'] = label_encoder.transform(test_dataset['Activity'])

# Store the actual activity names for later use in visualizations
activity_names = label_encoder.classes_

# Splitting the dataset into features and target label
features = train_dataset.drop(['Activity', 'subject'], axis=1)
labels = train_dataset['Activity']
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, random_state=42)

# Data transformation with standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Feature selection using Random Forest importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
selector = SelectFromModel(rf, prefit=True)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)

# Hyperparameter tuning with grid search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_selected, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Evaluation on the validation set
y_val_pred = best_model.predict(X_val_selected)
y_val_pred_labels = label_encoder.inverse_transform(y_val_pred)  # Convert to original labels
print(classification_report(y_val, y_val_pred_labels))

# Confusion matrix for the validation set
val_conf_matrix = confusion_matrix(y_val, y_val_pred_labels)
sns.heatmap(val_conf_matrix, annot=True, fmt='g', xticklabels=activity_names, yticklabels=activity_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Preparing the test dataset
X_test = test_dataset.drop(['Activity', 'subject'], axis=1)
y_test = test_dataset['Activity']
X_test = scaler.transform(X_test)
X_test_selected = selector.transform(X_test)

# Final evaluation on the test set
y_test_pred = best_model.predict(X_test_selected)
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)  # Convert to original labels
print(classification_report(y_test, y_test_pred_labels))

# Confusion matrix for the test set
test_conf_matrix = confusion_matrix(y_test, y_test_pred_labels)
sns.heatmap(test_conf_matrix, annot=True, fmt='g', xticklabels=activity_names, yticklabels=activity_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Log the model
mlflow.sklearn.log_model(best_model, "final_random_forest_classifier")
