import pandas as pd

# Load the full dataset
full_dataset = pd.read_csv(r"C:\Users\user\Documents\COM774_CW2\production\train_dataset.csv")
full_dataset1 = pd.read_csv(r"C:\Users\user\Documents\COM774_CW2\production\test_dataset.csv")

# Extract approximately 1/10th of the dataset in random order
sampled_dataset = full_dataset.sample(frac=0.1, random_state=42)
sampled_dataset1 = full_dataset1.sample(frac=0.1, random_state=30)

# Save the sampled data to a new CSV file
sampled_file_path = 'C:/Users/user/Documents/COM774_CW2/production/sampled_train_dataset.csv'
sampled1_file_path = 'C:/Users/user/Documents/COM774_CW2/production/sampled_test_dataset.csv'
sampled_dataset.to_csv(sampled_file_path, index=False)
sampled_dataset1.to_csv(sampled1_file_path, index=False)
