
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/floodnet_cleaned.csv")

columns = [
    "Maximum Flood Depth (inches)",
    "Total Duration (minutes)",
    "Time to Drain From Peak (minutes)",
    "Duration of Flooding Greater Than 4 Inches (minutes)"
]

for column in columns:
    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column])

    plt.title(f"Outlier Detection: {column}")
    plt.xlabel(column)

    plt.tight_layout()

    filename = column.lower().replace(" ", "_")
    plt.savefig(f"reports/figures/{filename}_boxplot.png")

    plt.show()

print("Boxplots generated successfully!")
