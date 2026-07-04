import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# 1. Konfigurasi Halaman Dashboard
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# 2. Tajuk Utama (Bahasa Inggeris - Selari dengan laporan rakan anda)
st.title("🎓 Online Learning Student Performance & Retention Analytics")
st.markdown("This interactive dashboard displays student engagement and predicts potential dropout risks using our trained Random Forest model.")
st.write("---")

# 3. Muat Naik Fail Data & Model
@st.cache_data # Mengurangkan loading yang lama setiap kali klik button
def load_data():
    return pd.read_csv("online_learning_engagement_cleaned.csv")

df = load_data()

with open("student_dropout_model.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================================
# SEKSYEN 1: TOP-LEVEL KPI METRICS
# ==========================================
col1, col2, col3, col4 = st.columns(4)

total_students = len(df)
avg_quiz = df['avg_quiz_score'].mean()
avg_attendance = df['attendance_rate'].mean() * 100
dropout_rate = (df['dropout'].sum() / total_students) * 100

with col1:
    st.metric(label="Total Students Analyzed", value=f"{total_students:,}")
with col2:
    st.metric(label="Average Quiz Score", value=f"{avg_quiz:.2f}/100")
with col3:
    st.metric(label="Average Attendance Rate", value=f"{avg_attendance:.1f}%")
with col4:
    st.metric(label="Overall Dropout Rate", value=f"{dropout_rate:.2f}%", delta="-Target < 5%", delta_color="inverse")

st.write("---")

# ==========================================
# SEKSYEN 2: VISUALISASI INTERAKTIF (PLOTLY)
# ==========================================
st.subheader("📊 Interactive Data Exploration")

# Cipta susunan dua lajur untuk graf
graph_col1, graph_col2 = st.columns(2)

with graph_col1:
    st.markdown("#### Attendance vs Final Grade")
    # Penapis (Filter) interaktif untuk peranti
    device_filter = st.selectbox("Filter by Device Type:", options=['All'] + list(df['device_type'].unique()))
    
    if device_filter == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['device_type'] == device_filter]
        
    fig_scatter = px.scatter(
        filtered_df, 
        x="attendance_rate", 
        y="final_grade", 
        color="dropout",
        color_discrete_map={0: "#2ec4b6", 1: "#e71d36"},
        labels={"attendance_rate": "Attendance Rate", "final_grade": "Final Grade", "dropout": "Status (1=Dropout)"},
        title=f"Impact of Attendance on Grades (Device: {device_filter})"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with graph_col2:
    st.markdown("#### Engagement Score Distribution")
    # Penapis interaktif mengikut umur pelajar
    age_range = st.slider("Select Student Age Range:", int(df['age'].min()), int(df['age'].max()), (18, 45))
    age_filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    
    fig_box = px.box(
        age_filtered_df, 
        x="dropout", 
        y="engagement_score", 
        color="dropout",
        color_discrete_map={0: "#2ec4b6", 1: "#e71d36"},
        labels={"dropout": "Status (0=Retained, 1=Dropout)", "engagement_score": "Engagement Score"},
        title=f"Engagement Levels for Age {age_range[0]} to {age_range[1]}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.write("---")

# ==========================================
# SEKSYEN 3: RAMALAN RISIKO DROPOUT (REAL-TIME MODEL PREDICTION)
# ==========================================
st.subheader("🔮 Intelligent Student Dropout Predictor")
st.markdown("Input a student's metrics below to evaluate their risk level using the 98.59% accurate Machine Learning model.")

# Membina borang input yang kemas menggunakan susunan lajur
input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    attendance_rate = st.slider("Attendance Rate (0.00 to 1.00)", 0.0, 1.0, 0.85, step=0.01)
    engagement_score = st.slider("Engagement Score (0.0 to 10.0)", 0.0, 10.0, 5.5, step=0.1)
    study_hours_weekly = st.number_input("Weekly Study Hours", min_value=0.0, max_value=60.0, value=15.0)
    login_frequency_weekly = st.number_input("Weekly Login Frequency", min_value=0, max_value=50, value=10)

with input_col2:
    avg_quiz_score = st.slider("Average Quiz Score (0 to 100)", 0.0, 100.0, 70.0, step=0.5)
    quiz_attempts = st.number_input("Number of Quiz Attempts", min_value=0, max_value=20, value=4)
    avg_session_duration_min = st.number_input("Avg Session Duration (Minutes)", min_value=0.0, value=40.0)

with input_col3:
    video_watch_time_min = st.number_input("Video Watch Time (Minutes)", min_value=0.0, value=150.0)
    assignments_submitted = st.number_input("Assignments Submitted", min_value=0, max_value=30, value=5)
    forum_posts = st.number_input("Forum Posts", min_value=0, max_value=50, value=2)

# Butang untuk memproses ramalan
if st.button("Analyze Student Retention Risk", type="primary"):
    # Susun input mengikut urutan 10 ciri (features) asal sewaktu train model di Colab
    features_input = np.array([[
        study_hours_weekly, login_frequency_weekly, avg_session_duration_min,
        video_watch_time_min, assignments_submitted, forum_posts,
        quiz_attempts, avg_quiz_score, attendance_rate, engagement_score
    ]])
    
    # Menjalankan ramalan model
    prediction = model.predict(features_input)
    prediction_proba = model.predict_proba(features_input)[0][1] * 100
    
    # Paparan hasil keputusan ramalan kepada pengguna
    st.write("### Result Analysis:")
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk Alert!** The student is predicted to **DROPOUT**. (Probability Risk: {prediction_proba:.2f}%)")
        st.markdown("- **Recommendation:** Immediate academic counseling and personalized follow-up on engagement are highly advised.")
    else:
        st.success(f"✅ **Safe!** The student is predicted to be **RETAINED** in the platform. (Probability Risk: {prediction_proba:.2f}%)")
        st.markdown("- **Recommendation:** The student demonstrates positive engagement. Maintain current academic path.")
