
import pandas as pd
import numpy as np

# Load original FloodNet dataset
file_path = "data/raw/floodnet_street_flooding.csv"
df = pd.read_csv(file_path)

# Basic dataset information
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- FIRST FIVE ROWS ---")
print(df.head())
