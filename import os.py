import os 
# reduce TF/transformers noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from transformers import pipeline
import pyttsx3
import tempfile
import io
import random

# ---------------- Global Sidebar Style ----------------
st.markdown("""
<style>
/* Force sidebar yellow across all pages */
section[data-testid="stSidebar"] {
    background-color: #FFD700 !important;
}
section[data-testid="stSidebar"] > div {
    background-color: #FFD700 !important;
}
section[data-testid="stSidebar"] * {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

def set_motion_background(page_name="default"):
    def random_rgb_colors():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f"rgb({r},{g},{b})"

    # create random gradient colors
    colors = [random_rgb_colors() for _ in range(6)]
    gradient = ", ".join(colors)

    st.markdown(f"""
    <style>
    /* Main animated gradient background */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(-45deg, {gradient});
        background-size: 400% 400%;
        animation: gradientBG 30s ease infinite;
    }}
    @keyframes gradientBG {{
        0% {{background-position: 0% 50%;}}
        25% {{background-position: 50% 100%;}}
        50% {{background-position: 100% 50%;}}
        75% {{background-position: 50% 0%;}}
        100% {{background-position: 0% 50%;}}
    }}

    /* Boxed content styling */
    .main .block-container > div {{
        background-color: rgba(0,0,0,0.85) !important;
        color: white !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }}

    /* Transparent header */
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """, unsafe_allow_html=True)

def random_rgb_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"rgb({r},{g},{b})"

def random_colors(n):
    return [random_rgb_color() for _ in range(n)]

# ---------------- Session Setup ----------------
if "users" not in st.session_state:
    st.session_state["users"] = {"admin": "admin123"}  # default user
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None
if "students" not in st.session_state:
    st.session_state["students"] = pd.DataFrame(columns=[
        "student_id", "name", "program", "semester",
        "attendance", "avg_score", "fees_status", "mentor_id",
        # Core Student Data
        "registration_number", "contact_info", "degree", "batch", "counselor",
        # Academic Performance Data
        "assessment_scores", "subject_performance", "num_attempts", "gpa",
        # Attendance Data
        "daily_attendance", "subject_attendance", "attendance_timestamps", "attendance_trends",
        # Financial Data
        "payment_history", "outstanding_dues",
        # Behavioral/Engagement Indicators
        "library_usage", "online_activity", "assignment_patterns", "participation",
        # Administrative Data
        "guardian_contact", "age", "location", "background", "prev_academic_history", "scholarship_status"
    ])

# ---------------- Risk Engine ----------------
def compute_risk(df, att_thresh=60, score_thresh=50):
    if df.empty:
        return df

    df = df.copy()
    df["risk_points"] = 0
    df.loc[df["attendance"] < att_thresh, "risk_points"] += 1
    df.loc[df["avg_score"] < score_thresh, "risk_points"] += 1
    df.loc[df["fees_status"] != "paid", "risk_points"] += 1

    def risk_level(points):
        if points >= 2:
            return "High"
        elif points == 1:
            return "Medium"
        else:
            return "Low"

    df["risk_level"] = df["risk_points"].apply(risk_level)
    return df

# ---------------- AI Automation ----------------
@st.cache_resource
def get_ai_model():
    return pipeline("text-generation", model="gpt2")

# ---------------- TTS Engine ----------------
@st.cache_resource
def get_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[0].id)
    return engine

def generate_audio(text):
    engine = get_tts_engine()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        tmp_path = fp.name
    engine.save_to_file(text, tmp_path)
    engine.runAndWait()
    with open(tmp_path, "rb") as f:
        audio_bytes = io.BytesIO(f.read())
    audio_bytes.seek(0)
    return audio_bytes

def generate_intervention(student):
    model = get_ai_model()
    prompt = (
        f"Student {student['name']} (ID: {student['student_id']}) has "
        f"attendance {student['attendance']}%, average score {student['avg_score']}%, "
        f"fee status {student['fees_status']}. Suggest an intervention plan for the mentor."
    )
    output = model(
        prompt,
        max_new_tokens=250,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    return output[0]['generated_text']

# ---------------- Authentication Pages ----------------
def login_page():
    set_motion_background()
    st.markdown(
        """
        <style>
        .login-container {
            max-width: 450px;
            margin: auto;
            padding: 40px;
            background: rgba(255,255,255,0.85);
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }
        .login-title {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #4B0082;
            margin-bottom: 20px;
        }
        .stTextInput>div>div>input {
            height: 45px;
            font-size: 16px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            height: 45px;
            width: 100%;
            border-radius: 10px;
            font-size: 16px;
            margin-top: 10px;
        }
        .stButton>button:hover {
            opacity: 0.9;
            transform: scale(1.02);
        }
        .center-text {
            text-align: center;
            margin-top: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔑 Student Dropout Dashboard</div>', unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    # Login button (centered full width)
    if st.button("Login"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["logged_in"] = True
            st.session_state["current_user"] = username
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    # Row for Forgot Password (aligned right)
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Forgot Password"):
            st.session_state["auth_mode"] = "forgot"
            st.rerun()

    # Don't have an account line
    st.markdown('<p class="center-text">Don’t have an account?</p>', unsafe_allow_html=True)

    # Centered Signup button
    col3, col4, col5 = st.columns([1, 2, 1])
    with col4:
        if st.button("Sign Up"):
            st.session_state["auth_mode"] = "signup"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
def login_page():
    set_motion_background()
    st.markdown(
        """
        <style>
        .login-card {
            max-width: 400px;
            margin: auto;
            margin-top: 60px;
            padding: 40px 30px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }
        .login-title {
            text-align: center;
            font-size: 22px;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }
        .login-subtitle {
            text-align: center;
            font-size: 14px;
            color: #666;
            margin-bottom: 25px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 1px solid #ccc;
            height: 42px;
            font-size: 15px;
            padding-left: 10px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #434343, #000000);
            color: white;
            height: 45px;
            width: 100%;
            border-radius: 10px;
            font-size: 15px;
            margin-top: 15px;
        }
        .forgot-pass {
            text-align: right;
            font-size: 13px;
            margin-top: -10px;
            margin-bottom: 15px;
        }
        .forgot-pass a {
            color: #2575fc;
            text-decoration: none;
        }
        .forgot-pass a:hover {
            text-decoration: underline;
        }
        .signup-section {
            text-align: center;
            margin-top: 25px;
            font-size: 14px;
        }
        .signup-btn {
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            color: white;
            height: 40px;
            width: 100%;
            border-radius: 10px;
            font-size: 15px;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Access your dropout dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Sign in with email</div>', unsafe_allow_html=True)

    username = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Forgot password link
    # Forgot password button (right aligned)
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Forgot Password?", key="forgot_btn"):
            st.session_state["auth_mode"] = "forgot"
            st.rerun()

    if st.button("Login"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["logged_in"] = True
            st.session_state["current_user"] = username
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    # Signup section
    st.markdown('<div class="signup-section">Don’t have an account?</div>', unsafe_allow_html=True)
    if st.button("Sign Up", key="signup", help="Create a new account", type="primary"):
        st.session_state["auth_mode"] = "signup"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Signup Page ----------------
def signup_page():
    set_motion_background()
    st.title("📝 Sign Up")
    new_user = st.text_input("Choose a username")
    new_pass = st.text_input("Choose a password", type="password")
    if st.button("Create Account"):
        if new_user in st.session_state["users"]:
            st.error("Username already exists!")
        else:
            st.session_state["users"][new_user] = new_pass
            st.success("Account created! Please log in.")
            st.session_state["auth_mode"] = "login"
            st.rerun()
    if st.button("Back to Login"):
        st.session_state["auth_mode"] = "login"
        st.rerun()

# ---------------- Forgot Password ----------------
def forgot_password_page():
    set_motion_background()
    st.markdown(
        """
        <style>
        .forgot-card {
            max-width: 400px;
            margin: auto;
            margin-top: 60px;
            padding: 40px 30px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }
        .forgot-title {
            text-align: center;
            font-size: 22px;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }
        .forgot-subtitle {
            text-align: center;
            font-size: 14px;
            color: #666;
            margin-bottom: 25px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 1px solid #ccc;
            height: 42px;
            font-size: 15px;
            padding-left: 10px;
        }
        .stButton>button {
            display: block;
            margin: auto;
            background: linear-gradient(90deg, #2575fc, #6a11cb);
            color: white;
            height: 42px;
            width: 180px;
            border-radius: 10px;
            font-size: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="forgot-card">', unsafe_allow_html=True)
    st.markdown('<div class="forgot-title">Forgot Password</div>', unsafe_allow_html=True)
    st.markdown('<div class="forgot-subtitle">Enter your email to reset your password</div>', unsafe_allow_html=True)

    email = st.text_input("Email")

    if st.button("Send Reset Link"):
        if email in st.session_state["users"]:
            st.success(f"Password reset link has been sent to {email}")
        else:
            st.error("Email not found in our records")

    if st.button("Back to Login"):
        st.session_state["auth_mode"] = "login"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Import Data ----------------
def import_data_page():
    set_motion_background()
    st.title("📥 Import Student Data")
    st.markdown("Upload CSV or Excel files to import student data. Data will be consolidated with existing records.")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_import = pd.read_csv(uploaded_file)
            else:
                df_import = pd.read_excel(uploaded_file)

            st.subheader("Preview of Imported Data")
            st.dataframe(df_import.head())

            # Ensure required columns exist, fill missing with NaN
            required_cols = st.session_state["students"].columns.tolist()
            for col in required_cols:
                if col not in df_import.columns:
                    df_import[col] = np.nan

            # Convert data types if possible
            df_import['attendance'] = pd.to_numeric(df_import['attendance'], errors='coerce')
            df_import['avg_score'] = pd.to_numeric(df_import['avg_score'], errors='coerce')
            df_import['gpa'] = pd.to_numeric(df_import['gpa'], errors='coerce')
            df_import['outstanding_dues'] = pd.to_numeric(df_import['outstanding_dues'], errors='coerce')
            df_import['age'] = pd.to_numeric(df_import['age'], errors='coerce')

            if st.button("Import Data"):
                # Merge with existing data, update on student_id
                existing = st.session_state["students"]
                combined = pd.concat([existing, df_import], ignore_index=True)
                combined = combined.drop_duplicates(subset='student_id', keep='last')
                st.session_state["students"] = combined
                st.success(f"Imported {len(df_import)} records. Total students: {len(combined)}")

        except Exception as e:
            st.error(f"Error importing file: {e}")

    st.markdown("### Supported Formats")
    st.markdown("- **CSV**: Comma-separated values")
    st.markdown("- **Excel**: .xlsx or .xls files")
    st.markdown("- **Columns**: The file should include columns matching the student data schema. Missing columns will be filled with defaults.")

# ---------------- Manual Student Entry ----------------
def manual_input_page():
    set_motion_background()
    st.title("➕ Add Student Data")
    with st.form("student_form"):
        sid = st.text_input("Student ID")
        name = st.text_input("Name")
        program = st.selectbox("Program", ["BSc CS", "BA Eng", "BCom"])
        semester = st.selectbox("Semester", [1,2,3,4,5,6])
        attendance = st.slider("Attendance %", 0, 100, 75)
        score = st.slider("Average Score %", 0, 100, 60)
        fees = st.selectbox("Fees Status", ["paid","partial","unpaid"])
        mentor = st.text_input("Mentor ID", "Mentor_A")
        submitted = st.form_submit_button("Save Student")
        if submitted:
            new_row = pd.DataFrame([{
                "student_id": sid,
                "name": name,
                "program": program,
                "semester": semester,
                "attendance": attendance,
                "avg_score": score,
                "fees_status": fees,
                "mentor_id": mentor
            }])
            st.session_state["students"] = pd.concat(
                [st.session_state["students"], new_row], ignore_index=True
            )
            st.success(f"Student {name} added!")
    if not st.session_state["students"].empty:
        st.subheader("📋 Current Student Records")
        st.dataframe(st.session_state["students"])

# ---------------- Dashboard ----------------
def dashboard_page():
    dashboard_bg = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #ff512f, #dd2476, #1fa2ff, #12d8fa);
        background-size: 400% 400%;
        animation: dashGradient 20s ease infinite;
    }
    @keyframes dashGradient {
        0% {background-position: 0% 50%;}
        25% {background-position: 50% 100%;}
        50% {background-position: 100% 50%;}
        75% {background-position: 50% 0%;}
        100% {background-position: 0% 50%;}
    }
    .main .block-container > div {
        background-color: rgba(0,0,0,0.8) !important;
        color: white !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """
    st.markdown(dashboard_bg, unsafe_allow_html=True)

    st.title("📊 Dropout Dashboard")

    df = compute_risk(st.session_state["students"])

    if df.empty:
        st.warning("No student data available yet. Add some manually!")
        return

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("👨‍🎓 Total Students", len(df))
    col2.metric("⚠️ At-Risk Students", (df["risk_level"]=="High").sum())
    col3.metric("📉 Avg Attendance %", round(df["attendance"].mean(),1))

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "👨‍🎓 Students", "🚨 Alerts",
        "⚙️ Threshold Preview", "🛠 Developer Notes"
    ])

    with tab1:
        st.subheader("Risk Distribution")
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        fig = px.bar(
            risk_counts,
            x="risk_level",
            y="count",
            color="risk_level",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Fee Status Distribution")
        fee_counts = df["fees_status"].value_counts().reset_index()
        fee_counts.columns = ["fees_status", "count"]
        fig2 = px.bar(
            fee_counts,
            x="fees_status",
            y="count",
            color="fees_status",
            color_discrete_sequence=px.colors.qualitative.Pastel1
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Attendance Trend Example")
        sample = df.sample(1).iloc[0]
        trend = pd.Series(
            np.clip(
                np.random.normal(sample["attendance"], 5, 8),
                30, 100
            ),
            index=pd.date_range(datetime.today()-timedelta(weeks=7), periods=8, freq="W")
        )
        st.line_chart(trend)

        st.subheader("📊 Cohort Radar Chart")
        metrics = {
            "Avg Attendance": df["attendance"].mean(),
            "Avg Score": df["avg_score"].mean(),
            "High Risk %": (df["risk_level"]=="High").mean() * 100,
            "Fee Defaulters %": (df["fees_status"]!="paid").mean() * 100,
        }
        radar_df = pd.DataFrame(dict(
            r=list(metrics.values()),
            theta=list(metrics.keys())
        ))
        fig = px.line_polar(radar_df, r="r", theta="theta", line_close=True)
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("All Students Data")
        st.dataframe(df)

        csv_all = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download All Students CSV",
            data=csv_all,
            file_name="students_data.csv",
            mime="text/csv"
        )

        st.markdown("### 👨‍🎓 Student Detail & Actions")
        sid = st.selectbox("Select Student", df["student_id"])
        student = df[df["student_id"] == sid].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Attendance %", student["attendance"])
        col2.metric("Avg Score", student["avg_score"])
        col3.metric("Fee Status", student["fees_status"])

        st.subheader("Mentor Notes (AI Generated)")

        if st.button("Generate AI Intervention Plan", key=f"ai_{student['student_id']}"):
            with st.spinner("Generating plan..."):
                ai_notes = generate_intervention(student)
                st.text_area("AI Suggested Notes", value=ai_notes, height=150)

                audio_file = generate_audio(ai_notes)
                st.audio(audio_file, format="audio/wav")
        else:
            st.text_area("AI Suggested Notes", placeholder="Click the button to generate an intervention plan...")

    with tab3:
        st.subheader("High Risk Students")
        risky = df[df["risk_level"]=="High"]
        if risky.empty:
            st.success("No students at high risk 🎉")
        else:
            st.dataframe(risky)

            csv_risky = risky.to_csv(index=False)
            st.download_button(
                label="⬇️ Download High Risk Students CSV",
                data=csv_risky,
                file_name="high_risk_students.csv",
                mime="text/csv"
            )

    with tab4:
        st.subheader("⚙️ Threshold Live Preview")
        att_thresh = st.slider("Min Attendance %", 0, 100, 60)
        score_thresh = st.slider("Min Avg Score %", 0, 100, 50)

        preview_df = compute_risk(st.session_state["students"], att_thresh, score_thresh)
        st.dataframe(preview_df[["student_id","name","attendance","avg_score","risk_level"]])

    with tab5:
        st.subheader("🛠 Developer Notes")
        st.markdown("""
        - Data ingestion supports manual entry (CSV upload can be added).
        - Risk engine uses thresholds (attendance, score, fees).
        - Adjust thresholds in **Threshold Live Preview** tab.
        - Extendable with ML models for predictive risk scoring.
        - Future scope: email/SMS alerts to mentors & parents.
        - Radar chart shows cohort health at a glance.
        - AI generates personalized intervention suggestions for at-risk students.
        """)

# ---------------- Main Router ----------------
def main():
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    if st.session_state.get("logged_in", False):
        # User is logged in → show main dashboard with sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Dashboard", "Add Student", "Import Data", "Logout"])
        if page == "Dashboard":
            dashboard_page()
        elif page == "Add Student":
            manual_input_page()
        elif page == "Import Data":
            import_data_page()
        elif page == "Logout":
            st.session_state["logged_in"] = False
            st.session_state["current_user"] = None
            st.session_state["auth_mode"] = "login"
            st.rerun()
    else:
        # User not logged in → show auth pages
        if st.session_state["auth_mode"] == "login":
            login_page()
        elif st.session_state["auth_mode"] == "signup":
            signup_page()
        elif st.session_state["auth_mode"] == "forgot":
            forgot_password_page()

if __name__ == "__main__":
    main()    