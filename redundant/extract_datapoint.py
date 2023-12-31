import pandas as pd

# Re-load the dataset as the previous code execution state was reset
#file_path = r"C:\Users\user\Documents\AI_MSc\COM774_CW2\test_dataset.csv
data = pd.read_csv(r"C:\Users\user\Documents\AI_MSc\COM774_CW2\test_dataset.csv")

# Extracting the 3rd row (index 2) data
third_row_data = data.iloc[2].tolist()

# Display the extracted data
third_row_data
