

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Scholarship Predictor",
    page_icon="🎓",
    layout="centered"
)


st.title("🎓 Scholarship Predictor")
st.write("Provide your details below to check eligible scholarships.")

MODEL_PATH = "scholarship_model.pkl"
LB_PATH = "label_binarizer.pkl"
clf, mlb = None, None

if os.path.exists(MODEL_PATH) and os.path.exists(LB_PATH):
    clf = joblib.load(MODEL_PATH)
    mlb = joblib.load(LB_PATH)
    
else:
    st.warning(" Model or label binarizer not found. App cannot predict scholarships.")

st.header("📋 Enter Your Details")

age = st.number_input("Age", min_value=15, max_value=40, value=20)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
state = st.text_input("State", "Tamil Nadu")
category = st.selectbox("Category", ["GEN", "OBC", "SC", "ST", "Minority"])
income = st.number_input("Annual Family Income (INR)", min_value=0, value=300000, step=5000)
disability = st.selectbox("Disability", [0, 1])
institution = st.selectbox("Institution Type", [
    "IIT", "NIT", "Central University", "State Government University", "Top Private", "Private"
])
program = st.selectbox("Program", ["UG", "PG"])
branch = st.text_input("Branch", "Computer Science")
year = st.number_input("Year of Study", min_value=1, max_value=4 if program=="UG" else 2, value=1)
final_year = 1 if (program == "UG" and year == 4) or (program == "PG" and year == 2) else 0
tenth = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=80.0)
twelfth = st.number_input("12th Percentage", min_value=0.0, max_value=100.0, value=80.0)
cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=10.0, value=8.0)
gate_score = 0
if program == "PG":
    gate_score = st.number_input("GATE Score (if applicable)", min_value=0, value=0)


if st.button("🔍 Predict Scholarships"):
    if clf and mlb:
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
        pred_labels = mlb.inverse_transform(pred)

        if pred_labels[0]:
            st.success(" Eligible Scholarships for you:")
            for s in pred_labels[0]:
                st.write(f" {s}")
        else:
            st.warning(" No matching scholarships found.")
    else:
        st.error("Model not loaded. Cannot perform prediction.")
