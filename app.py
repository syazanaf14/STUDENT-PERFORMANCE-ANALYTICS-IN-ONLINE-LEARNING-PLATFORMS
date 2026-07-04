import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="EduAnalytics Enterprise",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PREMIUM DARK THEME CSS (Business-Grade)
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Font & Background overrides */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
}

/* Hide default Streamlit top bar & footer for professional product feel */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 100% !important; }

/* Custom Enterprise Header */
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

/* Custom Executive Metrics Cards */
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
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #9ca3af;
    margin-bottom: 8px;
    font-weight: 500;
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

/* Section Labels & Interactive Containers */
.section-label {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #1D9E75;
    font-weight: 600;
    margin: 2rem 0 1rem 0;
}
.content-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 1.5rem;
}

/* Clean Input Form Overrides */
div[data-testid="stForm"] {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 24px !important;
}
</style>
""", unsafe_allow_html=True)

# Helper function to inject dark styling into Plotly figures
def apply_executive_theme(fig, height=350):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8e8e8', family='Inter'),
        margin=dict(l=40, r=20, t=40, b=40),
        height=height,
        xaxis=dict(gridcolor='#1f2937', zerolinecolor='#1f2937'),
        yaxis=dict(gridcolor='#1f2937', zerolinecolor='#1f2937')
    )
    return fig

# =====================================================
# DATA & MODEL LOADING
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("online_learning_engagement_cleaned.csv")

try:
    df = load_data()
    with open("student_dropout_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Sila pastikan fail dataset '.csv' dan fail model '.pkl' berada dalam repository GitHub anda.")

# =====================================================
# ENTERPRISE HEADER BAR
# =====================================================
st.markdown("""
<div class="dash-header">
    <div class="dash-title">🎓 EduAnalytics Enterprise</div>
    <div class="dash-tag">Live Predictive System v2.0</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# EXECUTIVE NAVIGATION CONTROL (Sidebar)
# =====================================================
with st.sidebar:
    st.markdown("<br><h3 style='color:#ffffff; letter-spacing:-0.5px;'>Control Panel</h3>", unsafe_allow_html=True)
    st.write("Sila pilih modul analisis:")
    page = st.radio(
        "Menu Navigasi:",
        ["📈 Platform Overview", "📊 Behavioral Analysis", "🔮 Machine Learning Predictor"],
        label_visibility="collapsed"
    )
    st.write("---")
    st.markdown("<p style='color:#6b7280; font-size:12px;'>Sistem ini dioptimumkan untuk membantu pihak pentadbir mengesan risiko keciciran pelajar secara masa nyata.</p>", unsafe_allow_html=True)

# =====================================================
# MODUL 1: PLATFORM OVERVIEW
# =====================================================
if page == "📈 Platform Overview":
    st.markdown("<div class='section-label'>Prestasi Keseluruhan Platform</div>", unsafe_allow_html=True)
    
    # Kiraan Metrik Utama
    total_students = len(df)
    avg_quiz = df['avg_quiz_score'].mean()
    avg_attendance = df['attendance_rate'].mean() * 100
    dropout_rate = (df['dropout'].sum() / total_students) * 100

    # Paparan Kad Gaya Business-Grade
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Jumlah Pelajar Dipantau</div><div class='metric-value'>{total_students:,}</div><div class='metric-sub'>Pengguna Aktif Semasa</div></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Purata Skor Kuiz</div><div class='metric-value'>{avg_quiz:.1f}/100</div><div class='metric-sub'>Tahap Kefahaman Akademik</div></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Kadar Kehadiran Purata</div><div class='metric-value'>{avg_attendance:.1f}%</div><div class='metric-sub'>Sasaran Institusi > 80%</div></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"<div class='metric-card critical'><div class='metric-label'>Kadar Risiko Keciciran</div><div class='metric-value'>{dropout_rate:.2f}%</div><div class='metric-sub' style='color:#dc3545;'>Kumpulan Sasaran Intervensi</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Data Mentah Sistem (Pemerhatian Rawak)</div>", unsafe_allow_html=True)
    st.markdown("<div class='content-box'>", unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# MODUL 2: BEHAVIORAL ANALYSIS
# =====================================================
elif page == "📊 Behavioral Analysis":
    st.markdown("<div class='section-label'>Analisis Gelagat & Tabiat Pelajar</div>", unsafe_allow_html=True)
    
    # Dropdown Filter yang Minimalis
    st.markdown("<div class='content-box'>", unsafe_allow_html=True)
    device_list = ['Semua Jenis Peranti'] + list(df['device_type'].unique())
    selected_device = st.selectbox("Tapis Data Mengikut Peranti Akses Pelajar:", device_list)
    st.markdown("</div>", unsafe_allow_html=True)
    
    filtered_df = df if selected_device == 'Semua Jenis Peranti' else df[df['device_type'] == selected_device]
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.write("**Hubungan Antara Kehadiran Dengan Gred Akhir**")
        fig_scatter = px.scatter(
            filtered_df, x="attendance_rate", y="final_grade", color="dropout",
            color_discrete_map={0: "#1D9E75", 1: "#dc3545"},
            labels={"attendance_rate": "Kadar Kehadiran (0.0 - 1.0)", "final_grade": "Gred Akhir Pelajar", "dropout": "Status Keciciran"},
            opacity=0.5
        )
        fig_scatter.update_layout(legend=dict(title="Petunjuk:", labels=["Kekal (0)", "Tercicir (1)"]))
        st.plotly_chart(apply_executive_theme(fig_scatter), use_container_width=True)
        st.caption("Nota Perniagaan: Kelompok merah (pelajar tercicir) sangat padat di kawasan kehadiran bawah 70%. Kehadiran ialah penunjuk kritikal.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_g2:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.write("**Tahap Keaktifan (Engagement Score) Mengikut Status Pelajar**")
        fig_box = px.box(
            filtered_df, x="dropout", y="engagement_score", color="dropout",
            color_discrete_map={0: "#1D9E75", 1: "#dc3545"},
            labels={"dropout": "Status (0 = Kekal, 1 = Tercicir)", "engagement_score": "Skor Interaksi Platform"}
        )
        st.plotly_chart(apply_executive_theme(fig_box), use_container_width=True)
        st.caption("Nota Perniagaan: Pelajar yang kekal aktif (hijau) mempunyai median skor interaksi yang jauh lebih tinggi berbanding pelajar berisiko (merah).")
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# MODUL 3: MACHINE LEARNING PREDICTOR
# =====================================================
elif page == "🔮 Machine Learning Predictor":
    st.markdown("<div class='section-label'>Sistem Diagnosis & Ramalan Risiko AI</div>", unsafe_allow_html=True)
    st.write("Sila masukkan data aktiviti semasa seorang pelajar di bawah untuk menganalisis kebarangkalian keciciran secara automatik.")
    
    with st.form("prediction_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            st.write("**📊 Ukuran Fokus Utama**")
            attendance_rate = st.slider("Kadar Kehadiran Kelas (0.00 hingga 1.00)", 0.0, 1.0, 0.85, step=0.01)
            engagement_score = st.slider("Skor Interaksi Platform (0.0 hingga 10.0)", 0.0, 10.0, 6.0, step=0.1)
            study_hours_weekly = st.number_input("Jam Belajar Seminggu", 0.0, 60.0, 15.0)
            login_frequency_weekly = st.number_input("Kekerapan Log Masuk Seminggu", 0, 50, 10)
            
        with col_in2:
            st.write("**📝 Prestasi Kuiz & Sesi**")
            avg_quiz_score = st.slider("Purata Markah Kuiz (0 hingga 100)", 0.0, 100.0, 75.0, step=0.5)
            quiz_attempts = st.number_input("Jumlah Percubaan Jawab Kuiz", 0, 20, 4)
            avg_session_duration_min = st.number_input("Purata Tempoh Sesi (Minit)", 0.0, 200.0, 45.0)
            
        with col_in3:
            st.write("**📂 Aktiviti Tambahan**")
            video_watch_time_min = st.number_input("Jumlah Tontonan Video (Minit)", 0.0, 1000.0, 180.0)
            assignments_submitted = st.number_input("Jumlah Tugasan Dihantar", 0, 30, 6)
            forum_posts = st.number_input("Jumlah Hantaran di Forum Diskusi", 0, 50, 3)

        submit_btn = st.form_submit_button("Mula Analisis Risiko Pelajar", type="primary")

    if submit_btn:
        # Susunan input data diselaraskan dengan model Random Forest anda
        features_input = np.array([[
            study_hours_weekly, login_frequency_weekly, avg_session_duration_min,
            video_watch_time_min, assignments_submitted, forum_posts,
            quiz_attempts, avg_quiz_score, attendance_rate, engagement_score
        ]])
        
        prediction = model.predict(features_input)
        prediction_proba = model.predict_proba(features_input)[0][1] * 100
        
        st.markdown("<div class='section-label'>Laporan Diagnosis Model AI</div>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            # Gauge Chart Premium untuk pandangan korporat
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Kebarangkalian Risiko (%)", 'font': {'size': 14, 'color': '#9ca3af'}},
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
                height=220, margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            st.write("### Kesimpulan Keputusan:")
            if prediction[0] == 1:
                st.markdown(f"<h3 style='color:#dc3545; margin-top:0;'>⚠️ STATUS KRITIKAL: RISIKO TERCICIR TINGGI</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                * **Peratusan Bahaya:** `{prediction_proba:.2f}%`
                * **Sebab Utama Kejatuhan:** Corak input menunjukkan kadar kehadiran atau markah interaksi berada di bawah paras selamat.
                * **Tindakan Cadangan:** Sistem telah mengaktifkan pelonjak amaran. Pegawai akademik atau pensyarah penasihat wajib menghubungi pelajar ini dengan segera untuk sesi kaunseling atau intervensi akademik.
                """)
            else:
                st.markdown(f"<h3 style='color:#1D9E75; margin-top:0;'>✅ STATUS SELAMAT: PELAJAR KEKAL aktif</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                * **Peratusan Bahaya:** `{prediction_proba:.2f}%`
                * **Sebab Utama Kejayaan:** Pelajar mengekalkan konsistensi yang stabil dalam kehadiran kelas dan tugasan mingguan.
                * **Tindakan Cadangan:** Tiada tindakan segera diperlukan. Teruskan memantau secara berkala bagi mengekalkan momentum positif pelajar.
                """)
            st.markdown("</div>", unsafe_allow_html=True)
