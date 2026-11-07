
import pandas as pd
import numpy as np

file_path = "data/raw/floodnet_street_flooding.csv"
df = pd.read_csv(file_path)

numeric_columns = [
    "Maximum Flood Depth (inches)",
    "Time to Maximum Flood Depth (minutes)",
    "Time to Drain From Peak (minutes)",
    "Total Duration (minutes)",
    "Duration of Flooding Greater Than 4 Inches (minutes)",
    "Duration of Flooding Greater Than 12 Inches (minutes)",
    "Duration of Flooding Greater Than 24 Inches (minutes)"
]

print("--- NUMERIC DATA CLEANING ---")

for column in numeric_columns:
    # Remove thousands separators
    cleaned = df[column].astype("string").str.replace(",", "", regex=False)

    # Convert to numeric; genuine missing values remain NaN
    df[column] = pd.to_numeric(cleaned, errors="coerce")

    print(f"\n{column}")
    print("Data type:", df[column].dtype)
    print("Missing values:", df[column].isna().sum())

print("\n--- FINAL DATA CHECK ---")
print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("\nNumeric column types:")
print(df[numeric_columns].dtypes)

print("\nCleaning conversion completed.")

print("\n--- MISSING VALUE ANALYSIS ---")

print(df.isnull().sum())

print("\n--- MISSING VALUE PERCENTAGE ---")

missing_percentage = (df.isnull().sum() / len(df)) * 100

print(missing_percentage.round(2))

print("\n--- ROWS WITH MISSING VALUES ---")

missing_rows = df[df.isnull().any(axis=1)]

print("Number of rows:", len(missing_rows))
print(missing_rows[
    ["Time to Drain From Peak (minutes)"]
].head(10))

# Handle missing value using median imputation

column = "Time to Drain From Peak (minutes)"

median_value = df[column].median()

print("\n--- MISSING VALUE IMPUTATION ---")
print("Median value:", median_value)

df[column] = df[column].fillna(median_value)

print("Missing values after imputation:", df[column].isna().sum())

# Verify dataset
print("\nFinal shape:", df.shape)
print("Total missing values:", df.isnull().sum().sum())

# Convert datetime columns

datetime_columns = [
    "Flood Start Datetime (GMT)",
    "Flood End Datetime (GMT)"
]

for column in datetime_columns:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
        utc=True
    )

print("\n--- DATETIME TYPES ---")
print(df[datetime_columns].dtypes)

print("\nDatetime missing values:")
print(df[datetime_columns].isna().sum())

# Save cleaned dataset separately

output_path = "data/processed/floodnet_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Location:", output_path)
