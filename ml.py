
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_scholarship_model():
    """Train ML model for scholarship prediction"""
    print("🚀 STARTING ML MODEL TRAINING...")
    print("="*60)
    
    # Load the dataset
    try:
        df = pd.read_csv(r'C:\Users\Thrilok\ml_env\Lib\site-packages\ml_project\scholarship_dataset_with_caste_10000.csv')
        print("✅ Dataset loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Dataset file not found! Please check the file path.")
        return
    
    # Define features and targets
    feature_columns = ['course_level', 'category', 'gender', 'annual_income', 
                      'marks_10', 'marks_12', 'ug_cgpa', 'field_of_study', 
                      'disability', 'university_tier']

    target_columns = ['UG1_CentralSector', 'UG2_Reliance', 'UG3_PostMatric_SC_ST', 
                     'UG4_SwamiDayanand_Female', 'UG5_TopClass_SC', 'UG6_MeritMeans_OBC',
                     'UG7_AdityaBirla', 'UG8_Cummins_SC_ST_Engg', 'UG9_SitaramJindal', 
                     'UG10_Disability', 'PG1_AICTE_NSPS', 'PG2_ReliancePG', 
                     'PG3_PostMatric_SC_ST_OBC', 'PG4_NationalFellowship_OBC', 
                     'PG5_INSPIRE', 'PG6_ICSSR', 'PG7_CumminsPG_SC_ST', 'PG8_TataTrusts']

    # Prepare the data
    X = df[feature_columns].copy()
    y = df[target_columns]

    print(f"🎯 Features: {len(feature_columns)} columns")
    print(f"🎯 Targets: {len(target_columns)} scholarships")
    
    # Handle missing values in ug_cgpa (for UG students)
    X['ug_cgpa'] = X['ug_cgpa'].fillna(0)

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['course_level', 'category', 'gender', 'field_of_study', 'disability', 'university_tier']

    print("\n🔧 Preprocessing data...")
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"   Encoded: {col} → {len(le.classes_)} categories")

    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['annual_income', 'marks_10', 'marks_12', 'ug_cgpa']
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    print("   Scaled numerical features")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['course_level']
    )

    print(f"\n📈 Data split:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")

    # Train the model
    print("\n🤖 Training Multi-output Random Forest Classifier...")
    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
        )
    )
    
    model.fit(X_train, y_train)
    print("✅ Model training completed!")

    # Evaluate the model
    print("\n📊 MODEL EVALUATION")
    print("-" * 40)
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    h_loss = hamming_loss(y_test, y_pred)
    
    # Calculate accuracy for each scholarship
    accuracies = []
    for i, col in enumerate(target_columns):
        acc = accuracy_score(y_test[col], y_pred[:, i])
        accuracies.append(acc)
    
    avg_accuracy = np.mean(accuracies)
    
    print(f"🎯 Hamming Loss: {h_loss:.4f} (Lower is better)")
    print(f"🎯 Average Accuracy: {avg_accuracy:.2%}")
    print(f"🎯 Best Scholarship Accuracy: {max(accuracies):.2%}")
    print(f"🎯 Worst Scholarship Accuracy: {min(accuracies):.2%}")

    # Feature Importance
    print("\n🔍 FEATURE IMPORTANCE (Top 5)")
    print("-" * 40)
    feature_importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    for _, row in importance_df.head().iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.3f}")

    # Save the model and preprocessing objects
    print("\n💾 Saving model and preprocessing objects...")
    joblib.dump(model, 'scholarship_predictor_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    joblib.dump(target_columns, 'target_columns.pkl')
    
    print("✅ All files saved successfully!")
    print("\n📁 Saved files:")
    print("   - scholarship_predictor_model.pkl (Trained ML model)")
    print("   - label_encoders.pkl (Categorical encoders)")
    print("   - scaler.pkl (Feature scaler)")
    print("   - feature_columns.pkl (Feature names)")
    print("   - target_columns.pkl (Scholarship names)")

    return model, label_encoders, scaler, feature_columns, target_columns

def test_trained_model():
    """Test the trained model with sample students"""
    print("\n" + "="*60)
    print("🧪 TESTING TRAINED MODEL")
    print("="*60)
    
    try:
        # Load saved objects
        model = joblib.load('scholarship_predictor_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        target_columns = joblib.load('target_columns.pkl')
        
        print("✅ Model loaded successfully for testing!")
    except FileNotFoundError:
        print("❌ Model files not found. Please train the model first.")
        return

    def predict_scholarships_ml(student_data):
        """Pure ML prediction function"""
        # Create DataFrame from input
        input_df = pd.DataFrame([student_data])
        
        # Handle UG students (no CGPA)
        if student_data['course_level'] == 'UG':
            input_df['ug_cgpa'] = 0
        
        # Encode categorical variables
        categorical_columns = ['course_level', 'category', 'gender', 'field_of_study', 'disability', 'university_tier']
        for col in categorical_columns:
            if col in label_encoders:
                if student_data[col] in label_encoders[col].classes_:
                    input_df[col] = label_encoders[col].transform([student_data[col]])[0]
                else:
                    input_df[col] = -1
        
        # Scale numerical features
        numerical_columns = ['annual_income', 'marks_10', 'marks_12', 'ug_cgpa']
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
        
        # Ensure correct column order
        input_df = input_df[feature_columns]
        
        # Pure ML Prediction
        predictions = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)
        
        # Get results
        eligible_scholarships = []
        scholarship_details = []
        
        for i, (pred, target_col) in enumerate(zip(predictions, target_columns)):
            if pred == 1:
                prob = probabilities[i][0][1]
                eligible_scholarships.append(target_col)
                scholarship_details.append({
                    'scholarship': target_col,
                    'probability': f"{prob:.1%}",
                    'confidence': 'High' if prob > 0.7 else 'Medium' if prob > 0.5 else 'Low'
                })
        
        return eligible_scholarships, scholarship_details

    # Test Cases
    test_cases = [
        {
            'name': 'UG SC Student (Engineering, Low Income)',
            'data': {
                'course_level': 'UG', 'category': 'SC', 'gender': 'Male',
                'annual_income': 200000, 'marks_10': 92, 'marks_12': 88,
                'ug_cgpa': 0, 'field_of_study': 'Engineering',
                'disability': 'No', 'university_tier': 'Tier 2'
            }
        },
        {
            'name': 'PG OBC Student (Engineering, Medium Income)',
            'data': {
                'course_level': 'PG', 'category': 'OBC', 'gender': 'Female',
                'annual_income': 500000, 'marks_10': 85, 'marks_12': 82,
                'ug_cgpa': 8.2, 'field_of_study': 'Engineering',
                'disability': 'No', 'university_tier': 'Tier 1'
            }
        },
        {
            'name': 'UG GEN Student (High Merit, High Income)',
            'data': {
                'course_level': 'UG', 'category': 'GEN', 'gender': 'Female',
                'annual_income': 1900000, 'marks_10': 98, 'marks_12': 97,
                'ug_cgpa': 0, 'field_of_study': 'Engineering',
                'disability': 'No', 'university_tier': 'Tier 1'
            }
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🎓 TEST CASE {i}: {test_case['name']}")
        print("-" * 50)
        
        eligible, details = predict_scholarships_ml(test_case['data'])
        
        print(f"📊 ML Model predicts: {len(eligible)} scholarships")
        for detail in details:
            confidence_icon = "🟢" if detail['confidence'] == 'High' else "🟡" if detail['confidence'] == 'Medium' else "🔴"
            print(f"   {confidence_icon} {detail['scholarship']}")
            print(f"      Confidence: {detail['confidence']} ({detail['probability']})")

# Main execution
if __name__ == "__main__":
    print("🎯 SCHOLARSHIP PREDICTION ML MODEL TRAINER")
    print("="*60)
    
    # Train the model
    trained_objects = train_scholarship_model()
    
    # Test the model
    test_trained_model()
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📚 NEXT STEPS:")
    print("1. Use the saved model files for predictions")
    print("2. Run the prediction script for new students")
    print("3. The model uses PURE ML (Random Forest) - no rule-based logic!")
    print("\n🚀 Your ML project is ready!")
