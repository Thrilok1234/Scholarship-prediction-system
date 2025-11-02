import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
warnings.filterwarnings('ignore')

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

# --------------------------------------------------
# MODEL TRAINING FUNCTION
# --------------------------------------------------
def train_scholarship_model():
    st.info("🚀 Starting ML model training...")

    dataset_path = os.path.join(BASE_DIR, 'scholarship_dataset_with_caste_10000.csv')

    try:
        df = pd.read_csv(dataset_path)
        st.success("✅ Dataset loaded successfully!")
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at {dataset_path}")
        return

    feature_columns = ['course_level', 'category', 'gender', 'annual_income',
                       'marks_10', 'marks_12', 'ug_cgpa', 'field_of_study',
                       'disability', 'university_tier']
    target_columns = ['UG1_CentralSector', 'UG2_Reliance', 'UG3_PostMatric_SC_ST',
                      'UG4_SwamiDayanand_Female', 'UG5_TopClass_SC', 'UG6_MeritMeans_OBC',
                      'UG7_AdityaBirla', 'UG8_Cummins_SC_ST_Engg', 'UG9_SitaramJindal',
                      'UG10_Disability', 'PG1_AICTE_NSPS', 'PG2_ReliancePG',
                      'PG3_PostMatric_SC_ST_OBC', 'PG4_NationalFellowship_OBC',
                      'PG5_INSPIRE', 'PG6_ICSSR', 'PG7_CumminsPG_SC_ST', 'PG8_TataTrusts']

    X = df[feature_columns].copy()
    y = df[target_columns]

    label_encoders = {}
    categorical_columns = ['course_level', 'category', 'gender', 'field_of_study', 'disability', 'university_tier']

    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    numerical_columns = ['annual_income', 'marks_10', 'marks_12', 'ug_cgpa']
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['course_level']
    )

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                               max_depth=10, min_samples_split=5)
    )
    model.fit(X_train, y_train)
    st.success("✅ Model training completed!")

    # Save model files
    joblib.dump(model, os.path.join(BASE_DIR, 'scholarship_predictor_model.pkl'))
    joblib.dump(label_encoders, os.path.join(BASE_DIR, 'label_encoders.pkl'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
    joblib.dump(feature_columns, os.path.join(BASE_DIR, 'feature_columns.pkl'))
    joblib.dump(target_columns, os.path.join(BASE_DIR, 'target_columns.pkl'))
    st.success("💾 Model and preprocessing objects saved successfully!")

# --------------------------------------------------
# SCHOLARSHIP PREDICTOR CLASS
# --------------------------------------------------
class ScholarshipPredictor:
    def __init__(self):
        try:
            self.model = joblib.load(os.path.join(BASE_DIR, 'scholarship_predictor_model.pkl'))
            self.label_encoders = joblib.load(os.path.join(BASE_DIR, 'label_encoders.pkl'))
            self.scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
            self.feature_columns = joblib.load(os.path.join(BASE_DIR, 'feature_columns.pkl'))
            self.target_columns = joblib.load(os.path.join(BASE_DIR, 'target_columns.pkl'))
        except FileNotFoundError:
            st.error("❌ Model files not found. Please train the model first.")
            raise

        self.income_limits = {
            'UG1_CentralSector': 800000, 'UG2_Reliance': 1500000, 'UG3_PostMatric_SC_ST': 250000,
            'UG4_SwamiDayanand_Female': 600000, 'UG5_TopClass_SC': 600000,
            'UG6_MeritMeans_OBC': 1000000, 'UG9_SitaramJindal': 400000,
            'UG10_Disability': 250000, 'PG1_AICTE_NSPS': 800000,
            'PG3_PostMatric_SC_ST_OBC': 250000, 'PG4_NationalFellowship_OBC': 600000,
            'PG6_ICSSR': 600000, 'PG7_CumminsPG_SC_ST': 800000,
            'PG8_TataTrusts': 1000000
        }

    def preprocess_input(self, student_data):
        input_df = pd.DataFrame([student_data])
        if student_data['course_level'] == 'UG':
            input_df['ug_cgpa'] = 0

        categorical_columns = ['course_level', 'category', 'gender', 'field_of_study', 'disability', 'university_tier']
        for col in categorical_columns:
            if student_data[col] in self.label_encoders[col].classes_:
                input_df[col] = self.label_encoders[col].transform([student_data[col]])[0]
            else:
                input_df[col] = -1

        numerical_columns = ['annual_income', 'marks_10', 'marks_12', 'ug_cgpa']
        input_df[numerical_columns] = self.scaler.transform(input_df[numerical_columns])
        input_df = input_df[self.feature_columns]
        return input_df

    def predict(self, student_data):
        processed = self.preprocess_input(student_data)
        predictions = self.model.predict(processed)[0]
        probabilities = self.model.predict_proba(processed)
        results = []

        for i, (pred, target_col) in enumerate(zip(predictions, self.target_columns)):
            if pred == 1:
                prob = probabilities[i][0][1]
                confidence = 'High' if prob > 0.7 else 'Medium' if prob > 0.5 else 'Low'
                income_limit = self.income_limits.get(target_col, None)
                status = "Eligible"
                if income_limit and student_data['annual_income'] > income_limit:
                    status = "Income Too High"
                results.append({
                    'Scholarship': target_col,
                    'Probability': f"{prob:.1%}",
                    'Confidence': confidence,
                    'Status': status,
                    'Income Limit': f"₹{income_limit:,}" if income_limit else "No Limit"
                })
        return results

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(page_title="🎓 Scholarship Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Scholarship Predictor")

option = st.sidebar.radio("Choose Option", ["Predict Scholarships", "Train Model"])

# ---------------- TRAIN MODEL SECTION ----------------
if option == "Train Model":
    st.header("⚙️ Train Scholarship Model")
    if st.button("🚀 Start Training"):
        train_scholarship_model()

# ---------------- PREDICTION SECTION ----------------
elif option == "Predict Scholarships":
    st.header("📋 Enter Student Details")

    with st.form("input_form"):
        course_level = st.selectbox("Course Level", ["UG", "PG"])
        category = st.selectbox("Category", ["GEN", "OBC", "SC", "ST"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        annual_income = st.number_input("Annual Family Income (₹)", min_value=0, step=10000)
        marks_10 = st.slider("10th Marks (%)", 0, 100, 85)
        marks_12 = st.slider("12th Marks (%)", 0, 100, 85)
        ug_cgpa = 0
        if course_level == "PG":
            ug_cgpa = st.slider("UG CGPA (out of 10)", 0.0, 10.0, 8.0)
        field = st.selectbox("Field of Study", ["Engineering", "Medicine", "Sciences", "Arts", "Commerce"])
        disability = st.selectbox("Disability", ["No", "Yes"])
        university_tier = st.selectbox("University Tier", ["Tier 1", "Tier 2", "Tier 3"])
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        try:
            predictor = ScholarshipPredictor()
            student_data = {
                'course_level': course_level,
                'category': category,
                'gender': gender,
                'annual_income': annual_income,
                'marks_10': marks_10,
                'marks_12': marks_12,
                'ug_cgpa': ug_cgpa,
                'field_of_study': field,
                'disability': disability,
                'university_tier': university_tier
            }
            results = predictor.predict(student_data)

            eligible = [r for r in results if r["Status"] == "Eligible"]
            ineligible = [r for r in results if r["Status"] != "Eligible"]

            if eligible:
                st.success(f"✅ Eligible for {len(eligible)} Scholarships!")
                for r in eligible:
                    st.write(f"🎯 **{r['Scholarship']}** — {r['Confidence']} Confidence ({r['Probability']})")
                    if r['Income Limit'] != 'No Limit':
                        st.caption(f"💰 Income Limit: {r['Income Limit']}")

            if ineligible:
                st.warning("⚠️ Potentially Eligible (Income Too High):")
                for r in ineligible:
                    st.write(f"🔸 **{r['Scholarship']}** — {r['Confidence']} ({r['Probability']})")
                    st.caption(f"❌ Income exceeds limit ({r['Income Limit']})")

            if not eligible and not ineligible:
                st.error("❌ Not eligible for any scholarships in the database.")

        except Exception as e:
            st.error(f"Error: {e}")
