
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/processed/floodnet_cleaned.csv")

# Convert datetime columns
df["Flood Start Datetime (GMT)"] = pd.to_datetime(
    df["Flood Start Datetime (GMT)"], utc=True
)

df["Flood End Datetime (GMT)"] = pd.to_datetime(
    df["Flood End Datetime (GMT)"], utc=True
)

# 1. Extract date and time features
df["Flood Start Hour"] = df["Flood Start Datetime (GMT)"].dt.hour
df["Flood Start Day"] = df["Flood Start Datetime (GMT)"].dt.day
df["Flood Start Month"] = df["Flood Start Datetime (GMT)"].dt.month
df["Flood Start Weekday"] = df["Flood Start Datetime (GMT)"].dt.dayofweek

# 2. Create flood severity categories
def classify_severity(depth):
    if depth < 4:
        return "Low"
    elif depth < 12:
        return "Moderate"
    elif depth < 24:
        return "High"
    else:
        return "Severe"

df["Flood Severity"] = df[
    "Maximum Flood Depth (inches)"
].apply(classify_severity)

# 3. Display new features
print("\n--- NEW FEATURES ---")
print(df[[
    "Flood Start Hour",
    "Flood Start Day",
    "Flood Start Month",
    "Flood Start Weekday",
    "Flood Severity"
]].head())

print("\n--- SEVERITY DISTRIBUTION ---")
print(df["Flood Severity"].value_counts())

print("\n--- DATASET SHAPE ---")
print(df.shape)

# Save engineered dataset
df.to_csv(
    "data/processed/floodnet_featured.csv",
    index=False
)

print("\nFeature engineered dataset saved successfully!")
