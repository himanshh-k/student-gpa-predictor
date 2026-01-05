import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# Load trained models
# ===========================
reg_model = joblib.load('models/gpa_regressor.pkl')   # Regression
knn_model = joblib.load('models/knn_model.pkl')       # KNN
kmeans_model = joblib.load('models/kmeans.pkl') # KMeans

# ===========================
# Helper: Predict GPA
# ===========================
def predict_gpa(model, features):
    X = np.array(features).reshape(1, -1)
    return model.predict(X)[0]

# ===========================
# Helper: Optimal Study Hours
# ===========================
def find_optimal_study_hours(model, base_feature_dict, features_list, 
                             study_col='StudyTimeWeekly', 
                             min_h=0, max_h=20, target_gpa=3.5):
    hrs = np.arange(min_h, max_h + 1, 1)
    preds = []
    for h in hrs:
        feat = base_feature_dict.copy()
        feat[study_col] = h
        x_row = np.array([feat[f] for f in features_list]).reshape(1, -1)
        p = model.predict(x_row)[0]
        preds.append(p)
    preds = np.array(preds)
    idx = np.where(preds >= target_gpa)[0]
    if idx.size > 0:
        return {'achievable': True, 'required_hours': int(hrs[idx[0]]), 'predicted_gpa': float(preds[idx[0]])}, hrs, preds
    else:
        max_idx = int(np.argmax(preds))
        return {'achievable': False, 'max_predicted_gpa': float(preds[max_idx]), 'hours_for_max': int(hrs[max_idx])}, hrs, preds

# ===========================
# Helper: Plot Optimal Study Hours
# ===========================
def plot_optimal_study_hours(hrs, preds, target_gpa, result):
    fig, ax = plt.subplots()
    ax.plot(hrs, preds, marker='o', color='b', label='Predicted GPA')
    if isinstance(result, dict) and result.get('achievable'):
        ax.axvline(result['required_hours'], color='g', linestyle='--', label=f"Required Hours: {result['required_hours']}")
        ax.axhline(target_gpa, color='orange', linestyle='--', label=f"Target GPA: {target_gpa}")
    else:
        ax.axvline(result[0]['hours_for_max'], color='r', linestyle='--', label=f"Max GPA at {result[0]['hours_for_max']} hrs")
    ax.set_xlabel("Study Hours per Week")
    ax.set_ylabel("Predicted GPA")
    ax.set_title("Optimal Study Hours vs Predicted GPA")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ===========================
# Helper: KNN Recommendation
# ===========================
def recommend_study_pattern(knn_model, input_features, csv_path="student_performance.csv", top_k=3):
    # Load data from CSV file
    df = pd.read_csv(csv_path)
    X_train = df.drop(['GPA', 'StudentID'], axis=1)
    
    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(np.array(input_features).reshape(1, -1), n_neighbors=top_k)
    similar_students = X_train.iloc[indices[0]]
    return similar_students

# ===========================
# Helper: Cluster Assignment
# ===========================
def get_cluster_category(cluster_label):
    if cluster_label == 0:
        return "High Achievers (High GPA:Absences ratio)"
    elif cluster_label == 1:
        return "Moderate Performers"
    else:
        return "Struggling Students (Low GPA:Absences ratio)"

# ===========================
# Streamlit Navigation
# ===========================
st.title("🎓 Student Performance Analyzer")
page = st.radio("Select a Page", 
                ["🏠 Home", "📈 GPA Prediction", "⏳ Optimal Study Hours", "📚 Study Pattern Recommendation", "🧩 Student Segmentation"])

st.divider()

# ===========================
# Page 1 - Home
# ===========================
if page == "🏠 Home":
    st.title("🎓 Student Performance Prediction Dashboard")
    st.write("Welcome! This app helps predict and analyze student GPA using regression, KNN, and KMeans clustering models.")
    st.markdown("""
    ### Available Features:
    1. **GPA Prediction** – Predict GPA based on study habits and demographics.  
    2. **Optimal Study Hours** – Get required study hours to reach your target GPA.  
    3. **Study Pattern Recommendation** – Find what top performers are doing differently.  
    4. **Student Segmentation** – See which cluster (High/Moderate/Struggling) a student belongs to.  
    """)
    st.success("Use the navigation above to explore different features.")

# ===========================
# Input Section (common for all pages except Home)
# ===========================
else:
    st.subheader("📝 Enter Student Information")
    
    age = st.number_input("Age", 15, 18, 16)
    gender = st.selectbox("Gender (0=Male, 1=Female)", [0, 1])
    ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3])
    parental_edu = st.selectbox("Parental Education (0=None, 1=HS, 2=College, 3=Bachelor, 4=Higher)", [0, 1, 2, 3, 4])
    study_hours = st.slider("Study Time Weekly (hrs)", 0, 20, 8)
    absences = st.slider("Absences", 0, 30, 2)
    tutoring = st.selectbox("Tutoring (0=No, 1=Yes)", [0, 1])
    parental_support = st.selectbox("Parental Support (0=None to 4=Very High)", [0, 1, 2, 3, 4])
    extracurricular = st.selectbox("Extracurricular (0=No, 1=Yes)", [0, 1])
    sports = st.selectbox("Sports (0=No, 1=Yes)", [0, 1])
    music = st.selectbox("Music (0=No, 1=Yes)", [0, 1])
    volunteering = st.selectbox("Volunteering (0=No, 1=Yes)", [0, 1])

    input_features = [age, gender, ethnicity, parental_edu, study_hours, absences, tutoring,
                      parental_support, extracurricular, sports, music, volunteering]

    st.divider()

# ===========================
# Page 2 - GPA Prediction
# ===========================
if page == "📈 GPA Prediction":
    st.title("📈 GPA Prediction")
    if st.button("Predict GPA"):
        predicted_gpa = predict_gpa(reg_model, input_features)
        st.success(f"🎯 Predicted GPA: **{predicted_gpa:.2f}**")

# ===========================
# Page 3 - Optimal Study Hours
# ===========================
elif page == "⏳ Optimal Study Hours":
    st.title("⏳ Find Your Optimal Study Hours")
    target_gpa = st.number_input("Enter Your Target GPA (2.0 - 4.0)", 2.0, 4.0, 3.5)
    if st.button("Find Optimal Hours"):
        feature_dict = {
            'Age': age,
            'Gender': gender,
            'Ethnicity': ethnicity,
            'ParentalEducation': parental_edu,
            'StudyTimeWeekly': study_hours,
            'Absences': absences,
            'Tutoring': tutoring,
            'ParentalSupport': parental_support,
            'Extracurricular': extracurricular,
            'Sports': sports,
            'Music': music,
            'Volunteering': volunteering
        }
        result, hrs, preds = find_optimal_study_hours(reg_model, feature_dict, list(feature_dict.keys()), target_gpa=target_gpa)

        if result.get('achievable'):
            st.success(f"✅ Target GPA {target_gpa} achievable with ~{result['required_hours']} hrs/week study.")
        else:
            st.warning(f"⚠️ Target GPA not achievable. Max Predicted GPA = {result['max_predicted_gpa']:.2f} at {result['hours_for_max']} hrs/week.")
        plot_optimal_study_hours(hrs, preds, target_gpa, result)

# ===========================
# Page 4 - Study Pattern Recommendation
# ===========================
elif page == "📚 Study Pattern Recommendation":
    st.title("📚 Study Pattern Recommendation (KNN)")
    st.info("Find similar students with higher GPA and learn from their habits.")
    if st.button("Find Recommendations"):
        st.write("🔍 Finding similar top-performing students...")
        similar_students = recommend_study_pattern(knn_model, input_features)
        st.dataframe(similar_students.head())

# ===========================
# Page 5 - Student Segmentation
# ===========================
elif page == "🧩 Student Segmentation":
    st.title("🧩 Student Segmentation (KMeans)")
    st.info("Cluster students into High Achievers, Moderate, and Struggling groups.")
    if st.button("Predict Cluster"):
        pred_gpa = predict_gpa(reg_model, input_features)
        cluster_features = [pred_gpa, study_hours, absences]
        cluster = kmeans_model.predict(np.array(cluster_features).reshape(1, -1))[0]
        cluster_name = get_cluster_category(cluster)
        st.success(f"🎯 The student belongs to cluster: **{cluster_name}**")