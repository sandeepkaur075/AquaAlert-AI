
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/processed/floodnet_cleaned.csv")

# Numeric columns
numeric_df = df.select_dtypes(include=np.number)

print("\n--- DESCRIPTIVE STATISTICS ---")
print(numeric_df.describe().round(2))

print("\n--- CORRELATION MATRIX ---")
correlation = numeric_df.corr()
print(correlation.round(2))

# 1. Maximum flood depth distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    df["Maximum Flood Depth (inches)"],
    bins=30,
    kde=True
)
plt.title("Distribution of Maximum Flood Depth")
plt.xlabel("Maximum Flood Depth (inches)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reports/figures/flood_depth_distribution.png")
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/figures/correlation_heatmap.png")
plt.show()

print("\nEDA completed. Graphs saved successfully.")
