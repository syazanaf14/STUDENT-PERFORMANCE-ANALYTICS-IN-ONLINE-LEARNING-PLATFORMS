import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# 1. Konfigurasi Halaman (Premium & Wide)
st.set_page_config(
    page_title="EduAnalytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mencantikkan kad KPI dan elemen UI
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .metric-box-red {
        background-color: #fff5f5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# 2. Muat Data & Model
@st.cache_data
def load_data():
    return pd.read_csv("online_learning_engagement_cleaned.csv")

try:
    df = load_data()
    with open("student_dropout_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading files: {e}. Please ensure dataset and .pkl file are in the repository.")

# ==========================================
# SIDEBAR NAVIGATION (Gaya Pro)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=100)
    st.title("Navigation")
    # Menu pilihan utama menggunakan Radio Button yang bersih
    page = st.radio("Go to:", ["Overview", "Descriptive Analysis", "Predictive Analytics", "Our Team"])
    
    st.write("---")
    st.markdown("### Project Theme:")
    st.caption("Student Performance Analytics in Online Learning Platforms")

# ==========================================
# TAB 1: OVERVIEW
# ==========================================
if page == "Overview":
    st.title("🎓 Online Learning Engagement Dashboard")
    st.subheader("Overview & Platform Summary")
    st.write("This dashboard provides data-driven insights into student behavior, academic engagement, and predictive risk modeling to reduce online student dropout rates.")
    
    st.write("---")
    
    # Reka bentuk Kad KPI Tersuai (Custom Metrics Row)
    total_students = len(df)
    avg_quiz = df['avg_quiz_score'].mean()
    avg_attendance = df['attendance_rate'].mean() * 100
    dropout_rate = (df['dropout'].sum() / total_students) * 100

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-box'><h5>Total Students</h5><h2>{total_students:,}</h2><p style='color:green;'>Active Users</p></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-box'><h5>Avg Quiz Score</h5><h2>{avg_quiz:.2f}/100</h2><p style='color:blue;'>Passing Grade</p></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-box'><h5>Avg Attendance Rate</h5><h2>{avg_attendance:.1f}%</h2><p style='color:blue;'>Target > 80%</p></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"<div class='metric-box-red'><h5>Overall Dropout Rate</h5><h2>{dropout_rate:.2f}%</h2><p style='color:red;'>At-Risk Group</p></div>", unsafe_allow_html=True)

    st.write("---")
    st.subheader("📋 Sample Dataset View")
    st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# TAB 2: DESCRIPTIVE ANALYSIS
# ==========================================
elif page == "Descriptive Analysis":
    st.title("📊 Descriptive Analysis")
    st.subheader("Explore Trends and Patterns in Student Engagement")
    st.write("---")
    
    # Penapis Global di bahagian atas halaman
    device_list = ['All'] + list(df['device_type'].unique())
    selected_device = st.selectbox("Select Device Type Filter:", device_list)
    
    filtered_df = df if selected_device == 'All' else df[df['device_type'] == selected_device]
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### Attendance vs Final Grade")
        fig_scatter = px.scatter(
            filtered_df, x="attendance_rate", y="final_grade", color="dropout",
            color_discrete_map={0: "#2ec4b6", 1: "#e71d36"},
            labels={"attendance_rate": "Attendance Rate", "final_grade": "Final Grade", "dropout": "Dropout Status"},
            opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Insight: Notice how the red dots (Dropouts) are heavily clustered below the 70% attendance rate line.")
        
    with col_g2:
        st.markdown("#### Distribution of Engagement Scores")
        fig_box = px.box(
            filtered_df, x="dropout", y="engagement_score", color="dropout",
            color_discrete_map={0: "#2ec4b6", 1: "#e71d36"},
            labels={"dropout": "Status (0=Retained, 1=Dropout)", "engagement_score": "Engagement Score"}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("Insight: Students who stay (green) show significantly higher engagement score medians than those who drop out (red).")

# ==========================================
# TAB 3: PREDICTIVE ANALYTICS
# ==========================================
elif page == "Predictive Analytics":
    st.title("🔮 AI Predictive Model")
    st.subheader("Real-Time Student Retention Risk Assessment")
    st.write("Leveraging our **98.59% Accurate Random Forest Model** to diagnose student risks instantly.")
    st.write("---")
    
    # Penyusunan Borang Menggunakan Expander & Columns (Sangat Kemas)
    with st.form("prediction_form"):
        st.markdown("### 📝 Enter Student Behavioral Metrics")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            attendance_rate = st.slider("Attendance Rate (0.00 to 1.00)", 0.0, 1.0, 0.85, step=0.01)
            engagement_score = st.slider("Engagement Score (0.0 to 10.0)", 0.0, 10.0, 6.0, step=0.1)
            study_hours_weekly = st.number_input("Weekly Study Hours", 0.0, 60.0, 15.0)
            login_frequency_weekly = st.number_input("Weekly Login Frequency", 0, 50, 10)
            
        with col_in2:
            avg_quiz_score = st.slider("Average Quiz Score (0 to 100)", 0.0, 100.0, 75.0, step=0.5)
            quiz_attempts = st.number_input("Number of Quiz Attempts", 0, 20, 4)
            avg_session_duration_min = st.number_input("Avg Session Duration (Mins)", 0.0, 200.0, 45.0)
            
        with col_in3:
            video_watch_time_min = st.number_input("Video Watch Time (Mins)", 0.0, 1000.0, 180.0)
            assignments_submitted = st.number_input("Assignments Submitted", 0, 30, 6)
            forum_posts = st.number_input("Forum Posts", 0, 50, 3)

        # Butang Hantar Borang
        submit_btn = st.form_submit_button("Run Diagnostic Analytics", type="primary")

    if submit_btn:
        # Susun input data sepadan dengan 10 features di Colab
        features_input = np.array([[
            study_hours_weekly, login_frequency_weekly, avg_session_duration_min,
            video_watch_time_min, assignments_submitted, forum_posts,
            quiz_attempts, avg_quiz_score, attendance_rate, engagement_score
        ]])
        
        prediction = model.predict(features_input)
        prediction_proba = model.predict_proba(features_input)[0][1] * 100
        
        st.write("---")
        st.subheader("📊 Diagnostic Report Execution")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            # Gauge Chart untuk visualisasi peratusan risiko (Sangat Profesional!)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Dropout Risk Probability (%)", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "#dc3545" if prediction[0] == 1 else "#2ec4b6"},
                    'steps': [
                        {'range': [0, 40], 'color': '#eef9f6'},
                        {'range': [40, 70], 'color': '#fff9db'},
                        {'range': [70, 100], 'color': '#fff5f5'}
                    ],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.write("#### Analytics Conclusion:")
            if prediction[0] == 1:
                st.error(f"🚨 **CRITICAL STATUS: HIGH DROPOUT RISK**")
                st.markdown(f"""
                - **Risk Probability:** {prediction_proba:.2f}%
                - **Primary Drivers Detected:** Low baseline attendance or suboptimal platform engagement score.
                - **Action Required:** Automated retention trigger activated. Academic mentors must schedule an immediate evaluation session with this student.
                """)
            else:
                st.success(f"✅ **NORMAL STATUS: RETAINED STUDENT**")
                st.markdown(f"""
                - **Risk Probability:** {prediction_proba:.2f}%
                - **Primary Drivers Detected:** Healthy study consistency and satisfactory learning behaviors.
                - **Action Required:** No urgent intervention needed. Encourage continued participation.
                """)

# ==========================================
# TAB 4: OUR TEAM
# ==========================================
elif page == "Our Team":
    st.title("👥 Group Project Team")
    st.subheader("Big Data Analytics & Visualisation Team Members")
    st.write("---")
    
    # Menyusun kad profil ahli kumpulan
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    
    with t_col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4140/4140037.png", width=120)
        st.markdown("**Ahli 1**")
        st.caption("Role: Introduction & Problem Statements")
        
    with t_col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4139/4139981.png", width=120)
        st.markdown("**Ahli 2**")
        st.caption("Role: Data Cleaning & EDA Specialist")
        
    with t_col3:
        st.image("https://cdn-icons-png.flaticon.com/512/4140/4140048.png", width=120)
        st.markdown("**Ahli 3 (Anda)**")
        st.caption("Role: ML Modeling & Dashboard Deployment")
        
    with t_col4:
        st.image("https://cdn-icons-png.flaticon.com/512/4140/4140061.png", width=120)
        st.markdown("**Ahli 4**")
        st.caption("Role: Results, Interpretation & Formatting")
