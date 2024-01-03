import pandas as pd


# Load the dataset

train_dataset = pd.read_csv(r"C:\Users\user\Documents\AI_MSc\COM774_CW2\train_dataset.csv")
test_dataset = pd.read_csv(r"C:\Users\user\Documents\AI_MSc\COM774_CW2\test_dataset.csv")
#df = pd.read_csv('path_to_your_dataset.csv')  # Replace with your data source

import pandas as pd

# Function to identify and remove duplicate columns
def remove_duplicate_columns(df):
    # Transpose the dataframe, drop duplicate rows (which were originally columns)
    # Then transpose it back to original form
    return df.T.drop_duplicates().T


# Original shapes of datasets
original_train_shape = train_dataset.shape
original_test_shape = test_dataset.shape

# Remove duplicate columns from both datasets
train_dataset_cleaned = remove_duplicate_columns(train_dataset)
test_dataset_cleaned = remove_duplicate_columns(test_dataset)

# Shapes of datasets after removing duplicates
cleaned_train_shape = train_dataset_cleaned.shape
cleaned_test_shape = test_dataset_cleaned.shape

# Output the shapes
print("Original Train Dataset Shape:", original_train_shape)
print("Cleaned Train Dataset Shape:", cleaned_train_shape)
print("Original Test Dataset Shape:", original_test_shape)
print("Cleaned Test Dataset Shape:", cleaned_test_shape)

# Check for missing data in both datasets
missing_data_train = train_dataset.isnull().sum()
missing_data_test = test_dataset.isnull().sum()

# Displaying the results
print("Missing Data in Train Dataset:\n", missing_data_train)
print("\nMissing Data in Test Dataset:\n", missing_data_test)


train_dataset_cleaned.to_csv(cleaned_train_dataset_path, index=False)
test_dataset_cleaned.to_csv(cleaned_test_dataset_path, index=False)



