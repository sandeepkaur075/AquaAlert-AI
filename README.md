# AquaAlert-AI – Urban Waterlogging Prediction Using Machine Learning

## Project Overview

AquaAlert-AI is a machine learning-based academic project that analyses historical urban street-flooding events and estimates flood severity and maximum flood depth.

The project uses the NYC FloodNet street-flooding dataset and provides an interactive Streamlit dashboard for exploring the data and testing trained machine learning models.

**Important:** This project analyses recorded flood events. It does not predict whether a flood will occur in the future.

## Objectives

- Analyse historical street-flooding data.
- Clean and preprocess the dataset.
- Perform Exploratory Data Analysis (EDA).
- Engineer useful time-based and severity features.
- Train and evaluate multiple machine learning classification models.
- Estimate maximum flood depth using Linear Regression.
- Develop an interactive Streamlit dashboard.

## Dataset

**Dataset:** FloodNet – Street Flooding Events Measured by FloodNet Sensors

**Source:** NYC Open Data

[Official Dataset](https://data.cityofnewyork.us/Environment/FloodNet-Street-Flooding-Events-Measured-by-FloodN/aq7i-eu5q)

The original dataset contains 3,581 records and 13 columns.

The project defines four severity categories based on maximum recorded flood depth:

| Severity | Maximum Flood Depth |
|---|---|
| Low | Below 4 inches |
| Moderate | 4 to below 12 inches |
| High | 12 to below 24 inches |
| Severe | 24 inches or above |

These are project-defined categories, not official FloodNet warning levels.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Streamlit
- Joblib
- Git and GitHub

## Machine Learning Models

### Classification

The following models were trained to classify the severity of recorded flooding events:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)
6. Naive Bayes

**Best accuracy:** Random Forest – 58.44%

**Best macro F1-score:** Decision Tree – 42.47%

### Regression

Linear Regression was used to estimate maximum flood depth.

| Metric | Result |
|---|---:|
| MAE | 3.0123 |
| MSE | 20.3840 |
| RMSE | 4.5149 |
| R² Score | 0.2134 |

The classification results show that predicting minority severity classes remains challenging.

## Project Structure

```text
AquaAlert-AI/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── screenshots/
├── src/
│   ├── data_cleaning.py
│   ├── data_inspection.py
│   ├── eda_analysis.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── numpy_practice.py
│   ├── outlier_analysis.py
│   └── preprocessing.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sandeepkaur075/AquaAlert-AI.git
cd AquaAlert-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare project data and models

The raw dataset and trained model files are excluded from GitHub. Place the downloaded CSV at:

`data/raw/floodnet_street_flooding.csv`

Run the project's preprocessing and model-training scripts to generate the processed data and model files before launching the dashboard.

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

## Dashboard Features

- Home dashboard with dataset statistics and visualisations.
- Flood severity prediction for supplied sensor and calendar inputs.
- Maximum flood depth estimation.
- Interactive charts for exploring historical flooding patterns.

## Limitations

- The dataset is specific to New York City.
- It contains recorded flood events and does not provide non-flood observations.
- The project does not perform flood occurrence prediction.
- Weather and rainfall variables are not included.
- Severity categories are defined for this project.
- Model performance, especially for Severe events, is limited by class imbalance and the small number of test samples.
- The dashboard does not use live sensor feeds.

## Project Report

The detailed academic project report is available in this repository:

`AquaAlert_Report.docx`

## Author

**Sandeep Kaur**  
M.Tech CSE – Data Science and Analytics  
Lovely Professional University, Punjab  
Semester 1 – 2026

## GitHub Repository

[https://github.com/sandeepkaur075/AquaAlert-AI](https://github.com/sandeepkaur075/AquaAlert-AI)

## Acknowledgement

Dataset provided by NYC Open Data through the FloodNet project. This project was developed for academic purposes.