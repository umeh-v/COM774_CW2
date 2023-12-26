import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--trainingdata', type=str, required=True, help='Dataset for training')
parser.add_argument('--testingdata', type=str, required=True, help='Dataset for testing')
args = parser.parse_args()

mlflow.autolog()
# Start an MLflow run
with mlflow.start_run():
    # Load datasets
    train_dataset = pd.read_csv(args.trainingdata)
    test_dataset = pd.read_csv(args.testingdata)

    train_dataset.head
    train_dataset.shape



    # Data preprocessing
    label_encoder = LabelEncoder()
    train_dataset['Activity'] = label_encoder.fit_transform(train_dataset['Activity'])
    test_dataset['Activity'] = label_encoder.transform(test_dataset['Activity'])

    # Splitting the dataset
    features = train_dataset.drop(['Activity', 'subject'], axis=1)
    labels = train_dataset['Activity']
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Data transformation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Model Selection and Hyperparameter Tuning
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Log parameters and best CV score
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # Plot and log feature importances
    feature_importances = best_model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), feature_importances[indices])
    plt.xticks(range(X_train.shape[1]), features.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    mlflow.log_artifact("feature_importances.png")

    # Model Evaluation on Validation Set
    y_val_pred = best_model.predict(X_val)
    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    mlflow.log_metric("val_accuracy", val_report['accuracy'])

    # Plot and log confusion matrix (Validation Set)
    val_conf_matrix = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(val_conf_matrix, annot=True, fmt='g')
    plt.title('Confusion Matrix for Validation Set')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig("val_confusion_matrix.png")
    mlflow.log_artifact("val_confusion_matrix.png")

    # Final Evaluation on Test Set
    X_test = test_dataset.drop(['Activity', 'subject'], axis=1)
    X_test = scaler.transform(X_test)
    y_test = test_dataset['Activity']
    y_test_pred = best_model.predict(X_test)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    test_report = classification_report(y_test, y_test_pred_labels, output_dict=True)
    mlflow.log_metric("test_accuracy", test_report['accuracy'])

    # Plot and log confusion matrix (Test Set)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_conf_matrix, annot=True, fmt='g')
    plt.title('Confusion Matrix for Test Set')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig("test_confusion_matrix.png")
    mlflow.log_artifact("test_confusion_matrix.png")

    # Log the final model
    mlflow.sklearn.log_model(best_model, "final_random_forest_classifier")
