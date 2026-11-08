
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AquaAlert AI",
    page_icon="🌊",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# Load trained models
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

# Dashboard heading
st.title("🌊 AquaAlert AI")
st.subheader("Urban Waterlogging Severity Analysis")

st.write(
    "An experimental Machine Learning dashboard "
    "for analysing recorded urban waterlogging events."
)

st.info(
    "This project classifies recorded flood-event severity. "
    "It does not forecast future flooding."
)

st.divider()

# Sidebar
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    ["Home", "Severity Prediction", "Flood Depth Estimation", "About"]
)

# Home page
if page == "Home":

    st.header("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Dataset Records", "3,581")

    with col2:
        st.metric("ML Models", "2")

    st.write(
        "AquaAlert AI uses historical FloodNet sensor records "
        "to study waterlogging severity and estimate flood depth."
    )

# Prediction inputs
elif page in ["Severity Prediction", "Flood Depth Estimation"]:

    st.header(page)

    st.write("Enter the event information below.")

    sensor_options = ["Select Sensor"]

    try:
        df = pd.read_csv(
            DATA_DIR / "floodnet_featured.csv"
        )

        sensor_options += sorted(
            df["Sensor Name"].dropna().astype(str).unique().tolist()
        )

    except Exception:
        st.error("Featured dataset could not be loaded.")
        st.stop()

    sensor_name = st.selectbox(
        "Sensor Name",
        sensor_options
    )

    sensor_id = st.selectbox(
    "Sensor ID",
    sorted(
        df.loc[
            df["Sensor Name"] == sensor_name,
            "Sensor ID"
        ].dropna().astype(str).unique().tolist()
    )
)
    

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

        if sensor_name == "Select Sensor":
            st.warning("Please select a sensor.")

        else:
            input_data = pd.DataFrame([{
                "Flood Start Hour": hour,
                "Flood Start Day": day,
                "Flood Start Month": month,
                "Flood Start Weekday": weekday,
                "Sensor Name": sensor_name,
                "Sensor ID": sensor_id
            }])

            if page == "Severity Prediction":

                prediction = rf_bundle["model"].predict(
                    rf_bundle["preprocessor"].transform(input_data)
                )[0]

                st.success(
                    f"Predicted Flood Severity: {prediction}"
                )

            else:

                prediction = lr_bundle["model"].predict(
                    lr_bundle["preprocessor"].transform(input_data)
                )[0]

                st.success(
                    f"Estimated Maximum Flood Depth: {prediction:.2f} inches"
                )

                if prediction < 0:
                    st.warning(
                        "The model produced a negative depth estimate. "
                        "This is outside the physically meaningful range."
                    )

# About page
elif page == "About":

    st.header("About AquaAlert AI")

    st.write(
        "This academic project uses the NYC FloodNet dataset "
        "and Machine Learning algorithms."
    )

    st.write("Models used:")

    st.write("- Random Forest Classifier")
    st.write("- Linear Regression")

    st.write(
        "The severity categories are project-defined using "
        "maximum flood depth."
    )

    st.warning(
        "The model is experimental and should not be used "
        "for real-world emergency decisions."
    )
