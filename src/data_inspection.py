
# ==========================================
# AQUAALERT-AI
# DAY 1 - PANDAS DATA INSPECTION
# ==========================================

import pandas as pd
import numpy as np

print("AQUAALERT-AI PROJECT")

# 1. Load original dataset
file_path = "data/raw/floodnet_street_flooding.csv"

df = pd.read_csv(file_path)

# 2. First five records
print("\n1. FIRST FIVE RECORDS")
print(df.head())

# 3. Dataset dimensions
print("\n2. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 4. Column names
print("\n3. COLUMN NAMES")
print(df.columns.tolist())

# 5. Data types
print("\n4. DATA TYPES")
print(df.dtypes)

# 6. Dataset information
print("\n5. DATASET INFORMATION")
df.info()

# 7. Statistical summary
print("\n6. STATISTICAL SUMMARY")
print(df.describe(include="all"))

# 8. Missing values
print("\n7. MISSING VALUES")
print(df.isnull().sum())

# 9. Duplicate records
print("\n8. DUPLICATE RECORDS")
print(df.duplicated().sum())

# 10. Numerical columns
print("\n9. NUMERICAL COLUMNS")
numeric_df = df.select_dtypes(include=np.number)
print(numeric_df.columns.tolist())

print("\nDATA INSPECTION COMPLETED!")

