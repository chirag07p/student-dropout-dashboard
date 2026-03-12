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

    colors = [random_rgb_colors() for _ in range(6)]
    gradient = ", ".join(colors)

    st.markdown(f"""
    <style>
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
    .main .block-container > div {{
        background-color: rgba(0,0,0,0.85) !important;
        color: white !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """, unsafe_allow_html=True)

def random_rgb_color():
    return f"rgb({random.randint(0, 255)},{random.randint(0, 255)},{random.randint(0, 255)})"

def random_colors(n):
    return [random_rgb_color() for _ in range(n)]

# ---------------- Session Setup ----------------
if "users" not in st.session_state:
    st.session_state["users"] = {"admin": "admin123"}  
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None
if "students" not in st.session_state:
    st.session_state["students"] = pd.DataFrame(columns=[
        "student_id", "name", "program", "semester", "attendance", "avg_score", "fees_status", "mentor_id",
        "registration_number", "contact_info", "degree", "batch", "counselor", "assessment_scores",
        "subject_performance", "num_attempts", "gpa", "daily_attendance", "subject_attendance",
        "attendance_timestamps", "attendance_trends", "payment_history", "outstanding_dues",
        "library_usage", "online_activity", "assignment_patterns", "participation", "guardian_contact",
        "age", "location", "background", "prev_academic_history", "scholarship_status"
    ])

# ---------------- Utility: Clean DataFrame ----------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe is Arrow-compatible by fixing column dtypes"""
    numeric_cols = ["attendance", "avg_score", "gpa", "num_attempts",
                    "outstanding_dues", "daily_attendance", "subject_attendance",
                    "participation", "age"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ensure batch is string
    if "batch" in df.columns:
        df["batch"] = df["batch"].astype(str)
    return df

# ---------------- Risk Engine ----------------
def compute_risk(students, att_thresh=60, score_thresh=50):
    df = clean_dataframe(pd.DataFrame(students))
    df["risk_level"] = "Low"
    df.loc[df["attendance"] < att_thresh, "risk_level"] = "Medium"
    df.loc[(df["attendance"] < att_thresh) & (df["avg_score"] < score_thresh), "risk_level"] = "High"
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
        f"Student {student['name']} (ID: {student.get('student_id','N/A')}) has "
        f"attendance {student.get('attendance','N/A')}%, avg score {student.get('avg_score','N/A')}%, "
        f"GPA {student.get('gpa','N/A')}, fee status {student.get('fees_status','N/A')}, "
        f"participation {student.get('participation','N/A')}%. "
        f"Suggest an intervention plan for the mentor."
    )
    output = model(prompt, max_new_tokens=250, do_sample=True, temperature=0.7, top_p=0.9)
    return output[0]['generated_text']

# ---------------- Login Page ----------------
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
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Access your dropout dashboard</div>', unsafe_allow_html=True)

    username = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["logged_in"] = True
            st.session_state["current_user"] = username
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Sign Up", key="signup"):
        st.session_state["auth_mode"] = "signup"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Dashboard ----------------
def dashboard_page():
    st.title("📊 Dropout Dashboard")
    df = compute_risk(st.session_state["students"])

    if df.empty:
        st.warning("No student data available yet. Add some manually!")
        return

    # Student detail section
    st.subheader("👨‍🎓 Student Detail & Actions")
    sid = st.selectbox("Select Student", df["student_id"].dropna().unique())
    if sid in df["student_id"].values:
        student = df[df["student_id"] == sid].iloc[0]
        st.metric("Attendance %", student.get("attendance", "N/A"))
        st.metric("GPA", student.get("gpa", "N/A"))

        if st.button("Generate Mentor Plan", key=f"ai_{sid}"):
            ai_notes = generate_intervention(student)
            st.text_area("AI Suggested Notes", value=ai_notes, height=150)
            st.audio(generate_audio(ai_notes), format="audio/wav")
    else:
        st.warning("⚠️ No student data available.")

# ---------------- Main Router ----------------
def main():
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"
    if st.session_state.get("logged_in", False):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Dashboard", "Add Student", "Import Data", "Logout"])
        if page == "Dashboard":
            dashboard_page()
        elif page == "Add Student":
            st.write("TODO: Add student page")
        elif page == "Import Data":
            st.write("TODO: Import data page")
        elif page == "Logout":
            st.session_state["logged_in"] = False
            st.session_state["auth_mode"] = "login"
            st.rerun()
    else:
        if st.session_state["auth_mode"] == "login":
            login_page()

if __name__ == "__main__":
    main()
