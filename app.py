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
# Helper: GPA Scale (4 → 10)
# ===========================
def gpa_4_to_10(gpa_4):
    return gpa_4 * 2.5

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
    preds_10 = preds * 2.5
    ax.plot(hrs, preds_10, marker='o', color='b', label='Predicted GPA (out of 10)')
    if isinstance(result, dict) and result.get('achievable'):
        ax.axvline(result['required_hours'], color='g', linestyle='--',
                   label=f"Required Hours: {result['required_hours']}")
        ax.axhline(target_gpa * 2.5, color='orange', linestyle='--',
                   label=f"Target GPA: {target_gpa * 2.5:.2f}/10")
    else:
        ax.axvline(result['hours_for_max'], color='r', linestyle='--',
                   label=f"Max GPA at {result['hours_for_max']} hrs")
    ax.set_xlabel("Study Hours per Week")
    ax.set_ylabel("Predicted GPA (out of 10)")
    ax.set_title("Optimal Study Hours vs Predicted GPA")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ===========================
# Helper: KNN Recommendation
# ===========================
def recommend_study_pattern(knn_model, input_features, csv_path="student_performance.csv", top_k=3):
    df = pd.read_csv(csv_path)
    data = df.drop(['GradeClass', 'StudentID'], axis=1)
    distances, indices = knn_model.kneighbors(np.array(input_features).reshape(1, -1), n_neighbors=top_k)
    similar_students = data.iloc[indices[0]]
    return similar_students

# ===========================
# Helper: Cluster Assignment
# ===========================
def get_cluster_category(cluster_label):
    if cluster_label == 2:
        return "High Achievers (High GPA, Low Absences)"
    elif cluster_label == 1:
        return "Moderate-to-Low Performers"
    else:
        return "Struggling Students (Low GPA, High Absences)"

# ===========================
# Streamlit Navigation
# ===========================
st.title("🎓 Student GPA Predictor")
page = st.radio(
    "Select a Page",
    ["🏠 Home", "📈 GPA Prediction", "⏳ Optimal Study Hours", "📚 Study Pattern Recommendation", "🧩 Student Segmentation"]
)

st.divider()

# ===========================
# Page 1 - Home
# ===========================
if page == "🏠 Home":
    st.title("🎓 Student GPA Prediction Dashboard")
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
# Input Section
# ===========================
else:
    st.subheader("📝 Enter Student Information")

    gender_map = {"Male": 0, "Female": 1}
    ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
    parental_edu_map = {"None": 0, "High School": 1, "Some College": 2, "Bachelor's": 3, "Higher": 4}
    parental_support_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
    binary_map = {"No": 0, "Yes": 1}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", 15, 18, 16)
    with col2:
        gender_label = st.selectbox("Gender", ["Male", "Female"])
        gender = gender_map[gender_label]
    with col3:
        ethnicity_label = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
        ethnicity = ethnicity_map[ethnicity_label]
    with col4:
        parental_edu_label = st.selectbox("Parental Education", ["None", "High School", "Some College", "Bachelor's", "Higher"])
        parental_edu = parental_edu_map[parental_edu_label]

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        study_hours = st.slider("Study Time Weekly (hrs)", 0, 20, 8)
    with col6:
        absences = st.slider("Absences", 0, 30, 2)
    with col7:
        tutoring_label = st.selectbox("Tutoring", ["No", "Yes"])
        tutoring = binary_map[tutoring_label]
    with col8:
        parental_support_label = st.selectbox("Parental Support", ["None", "Low", "Moderate", "High", "Very High"])
        parental_support = parental_support_map[parental_support_label]

    col9, col10, col11, col12 = st.columns(4)
    with col9:
        extracurricular_label = st.selectbox("Extracurricular Activities", ["No", "Yes"])
        extracurricular = binary_map[extracurricular_label]
    with col10:
        sports_label = st.selectbox("Sports", ["No", "Yes"])
        sports = binary_map[sports_label]
    with col11:
        music_label = st.selectbox("Music", ["No", "Yes"])
        music = binary_map[music_label]
    with col12:
        volunteering_label = st.selectbox("Volunteering", ["No", "Yes"])
        volunteering = binary_map[volunteering_label]

    input_features = [
        age, gender, ethnicity, parental_edu, study_hours, absences,
        tutoring, parental_support, extracurricular, sports, music, volunteering
    ]

    st.divider()

# ===========================
# Page 2 - GPA Prediction
# ===========================
if page == "📈 GPA Prediction":
    st.title("📈 GPA Prediction")
    if st.button("Predict GPA"):
        predicted_gpa_4 = predict_gpa(reg_model, input_features)
        predicted_gpa_10 = gpa_4_to_10(predicted_gpa_4)
        st.success(f"🎯 Predicted GPA: **{predicted_gpa_10:.2f} / 10**")

# ===========================
# Page 3 - Optimal Study Hours
# ===========================
elif page == "⏳ Optimal Study Hours":
    st.title("⏳ Find Your Optimal Study Hours")
    target_gpa_10 = st.number_input("Enter Your Target GPA (5.0 - 10.0)", 5.0, 10.0, 8.75)
    target_gpa = target_gpa_10 / 2.5

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

        result, hrs, preds = find_optimal_study_hours(
            reg_model, feature_dict, list(feature_dict.keys()), target_gpa=target_gpa
        )

        if result.get('achievable'):
            st.success(
                f"✅ Target GPA {target_gpa_10:.2f}/10 achievable with "
                f"~{result['required_hours']} hrs/week study."
            )
        else:
            st.warning(
                f"⚠️ Target GPA not achievable. Max Predicted GPA = "
                f"{gpa_4_to_10(result['max_predicted_gpa']):.2f}/10 "
                f"at {result['hours_for_max']} hrs/week."
            )

        plot_optimal_study_hours(hrs, preds, target_gpa, result)

# ===========================
# Page 4 - Study Pattern Recommendation
# ===========================
elif page == "📚 Study Pattern Recommendation":
    st.title("📚 Study Pattern Recommendation (KNN)")
    st.info("Find similar students with higher GPA and learn from their habits.")
    if st.button("Find Recommendations"):
        st.write("🔍 Finding similar performing students...")
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
