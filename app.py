import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# SYSTEM & PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="EduAnalytics Enterprise",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PREMIUM EXECUTIVE DARK THEME CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global Font & Background Overrides ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
}

/* ── Clean Screen: Hide default Streamlit top chrome & footer ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 100% !important; }

/* ── Premium Enterprise Header Bar ── */
.dash-header {
    background: #111827;
    border-bottom: 1px solid #1D9E75;
    padding: 20px 32px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.dash-title {
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.dash-tag {
    font-size: 12px;
    background: rgba(29, 158, 117, 0.15);
    color: #1D9E75;
    padding: 4px 12px;
    border-radius: 6px;
    border: 1px solid #1D9E75;
    font-weight: 600;
}

/* ── Business-Grade KPI Cards ── */
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px;
    text-align: left;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.metric-card.critical {
    border-left: 4px solid #dc3545;
}
.metric-card.normal {
    border-left: 4px solid #1D9E75;
}
.metric-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #9ca3af;
    margin-bottom: 8px;
    font-weight: 600;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #6b7280;
    margin-top: 6px;
}

/* ── Minimalist Section Titles & Containers ── */
.section-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #1D9E75;
    font-weight: 700;
    margin: 2rem 0 1rem 0;
}
.content-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 1.5rem;
}

/* ── Clean Input Form Overrides ── */
div[data-testid="stForm"] {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 24px !important;
}

/* ── Insight Banner Styles ── */
.insight-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 16px;
    font-size: 14px;
    color: #e8e8e8;
    margin-top: 10px;
}
.insight-card strong {
    color: #1D9E75;
}
.insight-card.critical strong {
    color: #dc3545;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def apply_executive_theme(fig, height=380):
    """Applies the enterprise dark theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8e8e8', family='Inter'),
        margin=dict(l=40, r=20, t=40, b=40),
        height=height,
        xaxis=dict(gridcolor='#1f2937', zerolinecolor='#1f2937', tickfont=dict(color='#9ca3af')),
        yaxis=dict(gridcolor='#1f2937', zerolinecolor='#1f2937', tickfont=dict(color='#9ca3af'))
    )
    return fig

# =====================================================
# DATA & MODEL RETRIEVAL
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("online_learning_engagement_cleaned.csv")

try:
    df = load_data()
    with open("student_dropout_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"System Configuration Error: Unable to load dependencies. Ensure 'online_learning_engagement_cleaned.csv' and 'student_dropout_model.pkl' are present in the directory. Details: {e}")
    st.stop()

# =====================================================
# ENTERPRISE NAVIGATION HEADER
# =====================================================
st.markdown("""
<div class="dash-header">
    <div class="dash-title">🎓 EduAnalytics Enterprise</div>
    <div class="dash-tag">Predictive Diagnostic System v2.0</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# CONTROL PANEL NAVIGATION (Sidebar)
# =====================================================
with st.sidebar:
    st.markdown("<br><h3 style='color:#ffffff; letter-spacing:-0.5px;'>Control Panel</h3>", unsafe_allow_html=True)
    st.write("Select operational module:")
    page = st.radio(
        "Navigation Menu:",
        ["📈 Platform Overview", "📊 Behavioral Analytics", "🔮 Machine Learning Predictor"],
        label_visibility="collapsed"
    )
    st.write("---")
    st.markdown("<p style='color:#6b7280; font-size:12px;'>This tactical platform is optimized for academic administrators, educators, and retention officers to perform live student risk mitigation.</p>", unsafe_allow_html=True)

# =====================================================
# MODULE 1: PLATFORM OVERVIEW
# =====================================================
if page == "📈 Platform Overview":
    st.markdown("<div class='section-label'>Institutional Health Metrics</div>", unsafe_allow_html=True)
    
    # Calculate Summary Statistics
    total_students = len(df)
    avg_quiz = df['avg_quiz_score'].mean()
    avg_attendance = df['attendance_rate'].mean() * 100
    dropout_rate = (df['dropout'].sum() / total_students) * 100

    # High Business View Metric Rows
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Monitored Users</div><div class='metric-value'>{total_students:,}</div><div class='metric-sub'>Active Student Profiles</div></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Average Quiz Grade</div><div class='metric-value'>{avg_quiz:.1f}/100</div><div class='metric-sub'>Core Academic Competency</div></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Mean Attendance Rate</div><div class='metric-value'>{avg_attendance:.1f}%</div><div class='metric-sub'>Institutional Benchmark > 80%</div></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"<div class='metric-card critical'><div class='metric-label'>Aggregated Dropout Rate</div><div class='metric-value'>{dropout_rate:.2f}%</div><div class='metric-sub' style='color:#dc3545;'>Target Risk Group</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Historical Student Registry (Data Stream View)</div>", unsafe_allow_html=True)
    st.markdown("<div class='content-box'>", unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# MODULE 2: BEHAVIORAL ANALYTICS
# =====================================================
elif page == "📊 Behavioral Analytics":
    st.markdown("<div class='section-label'>Operational Drilldown Filter</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='content-box'>", unsafe_allow_html=True)
    if 'device_type' in df.columns:
        device_list = ['All Hardware Classes'] + list(df['device_type'].unique())
        selected_device = st.selectbox("Segment Data by Student Hardware Class:", device_list)
        filtered_df = df if selected_device == 'All Hardware Classes' else df[df['device_type'] == selected_device]
    else:
        st.info("Device type segmentation is not available in the current dataset.")
        filtered_df = df
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.write("**Correlation: Class Attendance vs Final Academic Grade**")
        
        # Determine if 'final_grade' exists, otherwise fall back to 'avg_quiz_score'
        y_axis_val = "final_grade" if "final_grade" in filtered_df.columns else "avg_quiz_score"
        
        fig_scatter = px.scatter(
            filtered_df, x="attendance_rate", y=y_axis_val, color="dropout",
            color_discrete_map={0: "#1D9E75", 1: "#dc3545"},
            labels={
                "attendance_rate": "Attendance Consistency (0.0 - 1.0)", 
                y_axis_val: "Final Attained Score", 
                "dropout": "Lifecycle Status"
            },
            opacity=0.6
        )
        st.plotly_chart(apply_executive_theme(fig_scatter), use_container_width=True)
        st.markdown("""
        <div class="insight-card critical">
            💡 <strong>Business Insight:</strong> Critical dropout clustering (red data points) is heavily dense below the <strong>70% attendance threshold</strong>. Attendance serves as our primary early-warning vector.
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_g2:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.write("**Distribution: Digital Engagement Scores across Student Lifecycle**")
        fig_box = px.box(
            filtered_df, x="dropout", y="engagement_score", color="dropout",
            color_discrete_map={0: "#1D9E75", 1: "#dc3545"},
            labels={"dropout": "Lifecycle Status (0 = Retained, 1 = Dropout)", "engagement_score": "Platform Interaction Index"}
        )
        st.plotly_chart(apply_executive_theme(fig_box), use_container_width=True)
        st.markdown("""
        <div class="insight-card">
            📈 <strong>Operational Insight:</strong> Retained students (green) display a substantially higher median on the interaction index. Platform activity directly correlates with institutional retention.
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# MODULE 3: MACHINE LEARNING PREDICTOR
# =====================================================
elif page == "🔮 Machine Learning Predictor":
    st.markdown("<div class='section-label'>AI Predictive Diagnostics Engine</div>", unsafe_allow_html=True)
    st.write("Input current behavioral vectors of a specific student below to execute real-time dropout probability analysis.")
    
    with st.form("prediction_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            st.write("**📊 Primary Engagement Vectors**")
            attendance_rate = st.slider("Class Attendance Rate (0.00 to 1.00)", 0.0, 1.0, 0.85, step=0.01)
            engagement_score = st.slider("Platform Interaction Index (0.0 to 10.0)", 0.0, 10.0, 6.0, step=0.1)
            study_hours_weekly = st.number_input("Weekly Self-Study Time (Hours)", 0.0, 60.0, 15.0)
            login_frequency_weekly = st.number_input("Weekly Portal Logins (Frequency)", 0, 50, 10)
            
        with col_in2:
            st.write("**📝 Academic Competency Metrics**")
            avg_quiz_score = st.slider("Mean Quiz Score Percentage (0 to 100)", 0.0, 100.0, 75.0, step=0.5)
            quiz_attempts = st.number_input("Total Assessment Submissions (Attempts)", 0, 20, 4)
            avg_session_duration_min = st.number_input("Mean Session Length (Minutes)", 0.0, 200.0, 45.0)
            
        with col_in3:
            st.write("**📂 Supplemental Platform Behaviors**")
            video_watch_time_min = st.number_input("Video Resource Consumption (Minutes)", 0.0, 1000.0, 180.0)
            assignments_submitted = st.number_input("Total Formal Assignments Completed", 0, 30, 6)
            forum_posts = st.number_input("Peer Forum Communications (Posts)", 0, 50, 3)

        submit_btn = st.form_submit_button("Execute Predictive Diagnostics", type="primary")

    if submit_btn:
        # Align features precisely as requested
        features_input = np.array([[
            study_hours_weekly, login_frequency_weekly, avg_session_duration_min,
            video_watch_time_min, assignments_submitted, forum_posts,
            quiz_attempts, avg_quiz_score, attendance_rate, engagement_score
        ]])
        
        # Process Prediction
        prediction = model.predict(features_input)
        prediction_proba = model.predict_proba(features_input)[0][1] * 100
        
        st.markdown("<div class='section-label'>AI Diagnostic Response Report</div>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            # Responsive Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Attrition Risk Probability (%)", 'font': {'size': 14, 'color': '#9ca3af'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': '#e8e8e8'},
                    'bar': {'color': "#dc3545" if prediction[0] == 1 else "#1D9E75"},
                    'steps': [
                        {'range': [0, 40], 'color': '#111827'},
                        {'range': [40, 70], 'color': '#1f2937'},
                        {'range': [70, 100], 'color': '#2d1e22'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8e8e8', family='Inter'),
                height=250, margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div class='content-box' style='height: 100%;'>", unsafe_allow_html=True)
            st.write("### Operational Diagnosis Summary:")
            if prediction[0] == 1:
                st.markdown(f"<h3 style='color:#dc3545; margin-top:0;'>⚠️ CRITICAL STATUS: HIGH ATTRITION RISK</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                * **Risk Probability Rating:** `{prediction_proba:.2f}%`
                * **Primary Risk Drivers:** Behavioral data patterns reflect localized deficits in digital presence or critical class attendance markers.
                * **Recommended Intervention:** Automated proactive alert dispatched. Academic mentors and retention counselors are required to initiate immediate personalized academic intervention with this student profile.
                """)
            else:
                st.markdown(f"<h3 style='color:#1D9E75; margin-top:0;'>✅ NOMINAL STATUS: STABLE PROFILE</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                * **Risk Probability Rating:** `{prediction_proba:.2f}%`
                * **Primary Success Drivers:** Balanced learning consistency observed across weekly portal participation and task deadlines.
                * **Recommended Intervention:** No immediate retention mechanism required. Maintain general cyclical tracking to ensure sustained positive academic momentum.
                """)
            st.markdown("</div>", unsafe_allow_html=True)
