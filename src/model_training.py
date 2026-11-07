import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# Load featured dataset
df = pd.read_csv("data/processed/floodnet_featured.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Define target
y = df["Flood Severity"]

# Select input features
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTarget distribution:")
print(y.value_counts())
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

# Combine both pipelines
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed!")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
from imblearn.over_sampling import RandomOverSampler

# Balance training data only
oversampler = RandomOverSampler(random_state=42)

X_train_balanced, y_train_balanced = oversampler.fit_resample(
    X_train_processed,
    y_train
)

print("\nClass distribution after oversampling:")
print(y_train_balanced.value_counts())

print("\nTest data remains unchanged:")
print(y_test.value_counts())
# Convert data to dense format for Gaussian Naive Bayes
X_train_dense = X_train_balanced.toarray() if hasattr(X_train_balanced, "toarray") else X_train_balanced
X_test_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

# Define classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

results = []

# Train and evaluate each model
for name, model in models.items():

    print(f"\nTraining {name}...")

    if name == "Naive Bayes":
        model.fit(X_train_dense, y_train_balanced)
        predictions = model.predict(X_test_dense)
    else:
        model.fit(X_train_balanced, y_train_balanced)
        predictions = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Compare all models
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Select Random Forest
best_model = models["Random Forest"]

# Generate predictions
rf_predictions = best_model.predict(X_test_processed)

# Create confusion matrix
cm = confusion_matrix(y_test, rf_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Severity")
plt.ylabel("Actual Severity")

plt.tight_layout()
plt.savefig("reports/figures/random_forest_confusion_matrix.png")
plt.show()
# Detailed classification report
print("\n===== RANDOM FOREST CLASSIFICATION REPORT =====")

print(classification_report(
    y_test,
    rf_predictions,
    labels=["Low", "Moderate", "High", "Severe"],
    zero_division=0
))
import os

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Save model comparison results
results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# Save Random Forest model and preprocessing pipeline together
joblib.dump(
    {
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_columns": numeric_features + categorical_features
    },
    "models/random_forest_model.joblib"
)

print("\nModel comparison saved successfully!")
print("Trained Random Forest model saved successfully!")

# LINEAR REGRESSION EXPERIMENT

from sklearn.base import clone

# Numerical target: Maximum Flood Depth
y_depth = df["Maximum Flood Depth (inches)"]

# Separate train-test split for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X,
    y_depth,
    test_size=0.20,
    random_state=42
)

# Create a fresh preprocessing pipeline
reg_preprocessor = clone(preprocessor)

X_reg_train_processed = reg_preprocessor.fit_transform(X_reg_train)
X_reg_test_processed = reg_preprocessor.transform(X_reg_test)

# Train Linear Regression
linear_model = LinearRegression()

linear_model.fit(
    X_reg_train_processed,
    y_reg_train
)

# Make predictions
reg_predictions = linear_model.predict(X_reg_test_processed)

# Evaluate model
mae = mean_absolute_error(y_reg_test, reg_predictions)
mse = mean_squared_error(y_reg_test, reg_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_reg_test, reg_predictions)

print("\n===== LINEAR REGRESSION RESULTS =====")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
# Save Linear Regression results
regression_results = pd.DataFrame([{
    "Model": "Linear Regression",
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2 Score": r2
}])

regression_results.to_csv(
    "reports/linear_regression_results.csv",
    index=False
)

# Save regression model and preprocessor
joblib.dump(
    {
        "model": linear_model,
        "preprocessor": reg_preprocessor,
        "feature_columns": numeric_features + categorical_features
    },
    "models/linear_regression_model.joblib"
)

print("\nLinear Regression results saved successfully!")
import os

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

regression_results = pd.DataFrame([{
    "Model": "Linear Regression",
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2 Score": r2
}])

regression_results.to_csv(
    "reports/linear_regression_results.csv",
    index=False
)

joblib.dump(
    {
        "model": linear_model,
        "preprocessor": reg_preprocessor,
        "feature_columns": numeric_features + categorical_features
    },
    "models/linear_regression_model.joblib"
)

print("\nLinear Regression results saved successfully!")
print("File: reports/linear_regression_results.csv")
print("File: models/linear_regression_model.joblib")