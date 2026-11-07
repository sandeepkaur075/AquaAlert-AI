
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load featured dataset
df = pd.read_csv("data/processed/floodnet_featured.csv")

# Target: project-defined flood severity
y = df["Flood Severity"]

# Use only information available at event start.
# Exclude event outcomes and post-event measurements.
numeric_features = [
    "Flood Start Hour",
    "Flood Start Day",
    "Flood Start Month",
    "Flood Start Weekday"
]

categorical_features = [
    "Sensor Name",
    "Sensor ID"
]

X = df[numeric_features + categorical_features].copy()

# Train-test split with class stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Numeric preprocessing
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data without fitting on it
X_test_processed = preprocessor.transform(X_test)
# Handle class imbalance on training data only
oversampler = RandomOverSampler(random_state=42)


X_train_balanced, y_train_balanced = oversampler.fit_resample(
    X_train_processed,
    y_train
)


print("\n--- BEFORE OVERSAMPLING ---")
print(y_train.value_counts())

print("\n--- AFTER OVERSAMPLING ---")
print(y_train_balanced.value_counts())

print("\nTest data remains unchanged:")
print(y_test.value_counts())

print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nPreprocessing completed successfully!")
