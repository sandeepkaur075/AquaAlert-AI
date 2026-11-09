
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="AquaAlert AI",
    page_icon="🌊",
    layout="wide"
)

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# Load Models
@st.cache_resource
def load_models():
    rf_bundle = joblib.load(
        MODEL_DIR / "random_forest_model.joblib"
    )

    lr_bundle = joblib.load(
        MODEL_DIR / "linear_regression_model.joblib"
    )

    return rf_bundle, lr_bundle


try:
    rf_bundle, lr_bundle = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# Load Dataset
try:
    df = pd.read_csv(
        DATA_DIR / "floodnet_featured.csv"
    )
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()


# Dashboard Heading
st.title("🌊 AquaAlert AI")
st.subheader("Urban Waterlogging Severity Analysis")

st.write(
    "An experimental Machine Learning dashboard "
    "for analysing recorded urban waterlogging events."
)

st.info(
    "This project analyses recorded flood events. "
    "It does not forecast future flooding."
)

st.divider()


# Sidebar Navigation
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Home",
        "Severity Prediction",
        "Flood Depth Estimation",
        "About"
    ]
)


# HOME PAGE
if page == "Home":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Average Flood Depth",
            f"{df['Maximum Flood Depth (inches)'].mean():.2f} inches"
        )

    with col3:
        st.metric(
            "Maximum Flood Depth",
            f"{df['Maximum Flood Depth (inches)'].max():.2f} inches"
        )

    st.divider()

    # Chart 1: Severity Distribution
    st.subheader("Flood Severity Distribution")

    severity_counts = (
        df["Flood Severity"]
        .value_counts()
        .reindex(["Low", "Moderate", "High", "Severe"])
        .fillna(0)
    )

    st.bar_chart(severity_counts)

    st.divider()

        # Chart 2: Flood Depth Distribution
    st.subheader("Maximum Flood Depth Distribution")

    depth_bins = pd.cut(
        df["Maximum Flood Depth (inches)"],
        bins=20
    ).value_counts().sort_index()

    depth_chart = pd.DataFrame({
        "Depth Range": depth_bins.index.astype(str),
        "Number of Events": depth_bins.values
    })

    st.bar_chart(
        depth_chart.set_index("Depth Range")
    )

    st.divider()
        # Chart 3: Flood Events by Hour
    st.subheader("Flood Events by Hour")

    hourly_events = (
        df["Flood Start Hour"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(hourly_events)

    st.divider()

    # Chart 4: Average Flood Depth by Severity
    st.subheader("Average Flood Depth by Severity")

    average_depth = (
        df.groupby("Flood Severity")
           [ "Maximum Flood Depth (inches)"
           ]
        .mean()
        .reindex(["Low", "Moderate", "High", "Severe"])
    )

    st.bar_chart(average_depth)

    st.divider()

# PREDICTION PAGES
elif page in [
    "Severity Prediction",
    "Flood Depth Estimation"
]:

    st.header(page)

    st.write("Enter the event information below.")

    # Sensor Selection
    sensor_options = sorted(
        df["Sensor Name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    sensor_name = st.selectbox(
        "Sensor Name",
        sensor_options
    )

    sensor_id_options = sorted(
        df.loc[
            df["Sensor Name"].astype(str) == sensor_name,
            "Sensor ID"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    sensor_id = st.selectbox(
        "Sensor ID",
        sensor_id_options
    )

    # Date and Time Inputs
    col1, col2 = st.columns(2)

    with col1:

        hour = st.number_input(
            "Flood Start Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        day = st.number_input(
            "Flood Start Day",
            min_value=1,
            max_value=31,
            value=15
        )

    with col2:

        month = st.number_input(
            "Flood Start Month",
            min_value=1,
            max_value=12,
            value=6
        )

        weekday = st.number_input(
            "Flood Start Weekday (0=Monday, 6=Sunday)",
            min_value=0,
            max_value=6,
            value=2
        )

    predict_button = st.button(
        "Generate Prediction",
        type="primary"
    )

    if predict_button:

        input_data = pd.DataFrame([{
            "Flood Start Hour": hour,
            "Flood Start Day": day,
            "Flood Start Month": month,
            "Flood Start Weekday": weekday,
            "Sensor Name": sensor_name,
            "Sensor ID": sensor_id
        }])

        if page == "Severity Prediction":

            transformed_input = (
                rf_bundle["preprocessor"].transform(
                    input_data
                )
            )

            prediction = (
                rf_bundle["model"].predict(
                    transformed_input
                )[0]
            )

            st.success(
                f"Predicted Flood Severity: {prediction}"
            )

        else:

            transformed_input = (
                lr_bundle["preprocessor"].transform(
                    input_data
                )
            )

            prediction = (
                lr_bundle["model"].predict(
                    transformed_input
                )[0]
            )

            st.success(
                f"Estimated Maximum Flood Depth: "
                f"{prediction:.2f} inches"
            )

            if prediction < 0:
                st.warning(
                    "The model produced a negative depth estimate. "
                    "This is outside the physically meaningful range."
                )


# ABOUT PAGE
elif page == "About":

    st.header("About AquaAlert AI")

    st.write(
        "This academic project uses the NYC FloodNet dataset "
        "and Machine Learning algorithms."
    )

    st.subheader("Models Used")

    st.write("- Random Forest Classifier")
    st.write("- Linear Regression")

    st.subheader("Severity Categories")

    st.write("- Low: Below 4 inches")
    st.write("- Moderate: 4 to below 12 inches")
    st.write("- High: 12 to below 24 inches")
    st.write("- Severe: 24 inches or above")

    st.warning(
        "This is an experimental academic project. "
        "It should not be used for real-world emergency decisions."
    )
