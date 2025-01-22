import pandas as pd

# Read the parquet file
df = pd.read_parquet('train-00000-of-00001.parquet')

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Show the first few rows
print("\nFirst few rows:")
print(df.head())

# Display column names
print("\nColumns:")
print(df.columns.tolist())

# Show basic statistics
print("\nBasic statistics:")
print(df.describe())

# Show sample of data from each column
print("\nSample from each column:")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].head())
