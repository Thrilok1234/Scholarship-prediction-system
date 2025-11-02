import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -------------------- PATH SETUP --------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "scholarship_predictor_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
TARGET_PATH = os.path.join(BASE_DIR, "target_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "scholarship_dataset_with_caste_10000.csv")

# -------------------- STREAMLIT PAGE CONFIG --------------------
st.set_page_config(page_title="🎓 Scholarship Predictor", page_icon="🎯", layout="centered")

st.title("🎓 Scholarship Predictor")
st.write("Predict possible scholarships based on student background.")

# -------------------- TRAINING FUNCTION --------------------
def train_scholarship_model():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset file not found. Please make sure 'scholarship_dataset_with_caste_10000.csv' is uploaded.")
        return

    df = pd.read_csv(DATA_PATH)

    # Ensure necessary columns exist
    required_cols = ['Marks', 'Category', 'Achievements', 'Field', 'Disability', 'University_Tier', 'Scholarship']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Dataset missing columns: {missing}")
        return

    X = df[['Marks', 'Category', 'Achievements', 'Field', 'Disability', 'University_Tier']]
    y = df[['Scholarship']]

    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    y_encoder = LabelEncoder()
    y['Scholarship'] = y_encoder.fit_transform(y['Scholarship'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_scaled, y['Scholarship'])

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATURE_PATH)
    joblib.dump(list(y_encoder.classes_), TARGET_PATH)

    st.success("✅ Model trained and saved successfully!")

# -------------------- LOAD MODEL --------------------
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURE_PATH)
        target_classes = joblib.load(TARGET_PATH)
        return model, label_encoders, scaler, feature_columns, target_classes
    except Exception as e:
        st.error("❌ Model files not found. Please train the model first.")
        st.error(e)
        return None, None, None, None, None

# -------------------- SIDEBAR --------------------
option = st.sidebar.radio("Navigation", ["Predict Scholarships", "Train Model"])

# -------------------- PREDICTION PAGE --------------------
if option == "Predict Scholarships":
    model, label_encoders, scaler, feature_columns, target_classes = load_model()

    st.header("🔍 Enter Student Details")

    marks = st.slider("Marks (%)", 0, 100, 75)
    category = st.selectbox("Category", ["General", "OBC", "SC", "ST"])
    achievements = st.selectbox("Achievements", ["None", "District", "State", "National"])
    field = st.selectbox("Field of Study", ["Engineering", "Science", "Commerce", "Arts", "Medical"])
    disability = st.selectbox("Disability", ["No", "Yes"])
    university = st.selectbox("University Tier", ["Tier 1", "Tier 2", "Tier 3"])

    if st.button("🔮 Predict"):
        if not model:
            st.warning("Train the model first!")
        else:
            data = pd.DataFrame([{
                "Marks": marks,
                "Category": category,
                "Achievements": achievements,
                "Field": field,
                "Disability": disability,
                "University_Tier": university
            }])

            # Encode categorical features
            for col in label_encoders:
                data[col] = label_encoders[col].transform(data[col])

            # Scale
            data_scaled = scaler.transform(data[feature_columns])

            # Predict
            pred = model.predict(data_scaled)
            scholarship = target_classes[pred[0]]

            st.success(f"🎉 Eligible Scholarship: **{scholarship}**")

# -------------------- TRAIN MODEL PAGE --------------------
elif option == "Train Model":
    st.header("⚙️ Train Scholarship Model")
    st.write("Upload or confirm dataset and train the model.")

    if st.button("🚀 Train Model"):
        train_scholarship_model()
