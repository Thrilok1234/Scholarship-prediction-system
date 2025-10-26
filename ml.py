import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer

# === Streamlit App Title ===
st.title("🎓 Scholarship Eligibility Prediction")
st.write("Enter your details below to check which scholarships you are eligible for.")

# === STEP 1: Load Dataset Automatically ===
DATA_PATH = r"Scholarship_dataset_final.csv"

df = pd.read_csv(DATA_PATH)
df["gate_score"] = pd.to_numeric(df["gate_score"], errors="coerce").fillna(0)

# === STEP 2: Prepare Features and Labels ===
X = df.drop(columns=["id", "name", "eligible_scholarships"])
y_labels = df["eligible_scholarships"].fillna("").apply(lambda x: x.split(";") if x else [])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_labels)
joblib.dump(mlb, "label_binarizer.pkl")

# === STEP 3: Pipelines for Preprocessing ===
categorical = ["gender", "state", "category", "institution_type", "program", "branch"]
numeric = ["age", "family_income_annual_inr", "disability", "year", "final_year",
           "tenth_percent", "twelfth_percent", "ug_cgpa", "gate_score"]

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_pipeline, categorical),
        ("num", num_pipeline, numeric)
    ]
)

# === STEP 4: Model Training or Loading ===
model_file = "scholarship_model.pkl"

if os.path.exists(model_file):
    clf = joblib.load(model_file)
    st.success("✅ Existing model loaded successfully.")
else:
    st.info("🚀 Training new model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", MultiOutputClassifier(
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        ))
    ])
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_file)
    st.success("✅ Model trained and saved successfully.")

# === STEP 5: User Input Form ===
st.subheader("🧾 Enter Your Academic and Personal Details")

with st.form("scholarship_form"):
    age = st.number_input("Age", min_value=15, max_value=35, value=20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    state = st.text_input("State")
    category = st.selectbox("Category", ["GEN", "OBC", "SC", "ST", "Minority"])
    income = st.number_input("Annual Family Income (INR)", min_value=0)
    disability = st.selectbox("Disability", [0, 1])
    institution = st.selectbox("Institution Type", [
        "IIT", "NIT", "Central University", "State Government University",
        "Top Private", "Private"
    ])
    program = st.selectbox("Program", ["UG", "PG"])
    branch = st.text_input("Branch (e.g., Computer Science, Mechanical)")
    year = st.number_input("Year of Study", min_value=1, max_value=4 if program == "UG" else 2)
    final_year = 1 if (program == "UG" and year == 4) or (program == "PG" and year == 2) else 0
    tenth = st.number_input("10th Percentage", min_value=0.0, max_value=100.0)
    twelfth = st.number_input("12th Percentage", min_value=0.0, max_value=100.0)
    cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=10.0)
    gate_score = 0
    if program == "PG":
        gate_input = st.text_input("GATE Score (optional)", value="0")
        gate_score = int(gate_input) if gate_input.strip().isdigit() else 0

    submitted = st.form_submit_button("Predict Eligible Scholarships")

# === STEP 6: Prediction ===
if submitted:
    sample = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "state": state,
        "category": category,
        "family_income_annual_inr": income,
        "disability": disability,
        "institution_type": institution,
        "program": program,
        "branch": branch,
        "year": year,
        "final_year": final_year,
        "tenth_percent": tenth,
        "twelfth_percent": twelfth,
        "ug_cgpa": cgpa,
        "gate_score": gate_score
    }])

    pred = clf.predict(sample)
    mlb = joblib.load("label_binarizer.pkl")
    pred_labels = mlb.inverse_transform(pred)
    scholarships = pred_labels[0]

    st.subheader("🎯 Eligible Scholarships:")
    if scholarships:
        for s in scholarships:
            st.markdown(f"- {s}")
    else:
        st.info("No matching scholarships found for the entered details.")
