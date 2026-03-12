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
import base64

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
# ---------------- Global Logo for Logged-in Pages ----------------
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
        "student_id", "name", "program", "semester", "attendance", "avg_score", "fee_status", "mentor_id",
        "registration_number", "contact_info", "degree", "batch", "counselor", "assessment_scores",
        "subject_performance", "num_attempts", "gpa", "daily_attendance", "subject_attendance",
        "attendance_timestamps", "attendance_trends", "payment_history", "outstanding_dues",
        "library_usage", "online_activity", "assignment_patterns", "participation", "guardian_contact",
        "age", "location", "background", "prev_academic_history", "scholarship_status"
    ])

# ---------------- Risk Engine ----------------
def compute_risk(df: pd.DataFrame, att_thresh=75, score_thresh=50) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Ensure 'name' column exists
    if "name" not in df.columns:
        df["name"] = df["registration_number"]  # fallback

    # Thresholds (tune as needed)
    daily_attendance_thresh = att_thresh
    avg_score_thresh = score_thresh

    # Make sure required cols exist
    for col in ["daily_attendance", "avg_score"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Initialize risk level
    df["risk_level"] = "Low"

    # Medium risk: low attendance
    df.loc[df["daily_attendance"] < daily_attendance_thresh, "risk_level"] = "Medium"

    # High risk: low attendance + low score
    df.loc[
        (df["daily_attendance"] < daily_attendance_thresh) & 
        (df["avg_score"] < avg_score_thresh), 
        "risk_level"
    ] = "High"

    return df


# ---------------- AI Automation ----------------
@st.cache_resource
def get_ai_model():
    return pipeline("text-generation", model="distilgpt2")

# ---------------- TTS Engine ----------------
def get_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[0].id)
    return engine

def generate_audio(text: str):
    engine = get_tts_engine()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        tmp_path = fp.name
    # ensure file handle is closed before pyttsx3 writes to it
    engine.save_to_file(text, tmp_path)
    engine.runAndWait()
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(tmp_path)  # cleanup temp file
    return audio_bytes

# ---------------- Intervention ----------------
def generate_intervention(student):
    model = get_ai_model()
    prompt = (
        f"Student {student.get('name', 'Unknown')} (ID: {student.get('student_id', 'N/A')}) has "
        f"daily_attendance {student.get('daily_attendance', 'N/A')}%, subject_attendance {student.get('subject_attendance', 'N/A')}%, attendance_trends {student.get('attendance_trends', 'N/A')}, "
        f"average score {student.get('avg_score', 'N/A')}%, GPA {student.get('gpa', 'N/A')}, "
        f"fee status {student.get('fee_status', 'N/A')}. "
        f"Suggesting a brief intervention plan for the mentor to help this student improve. Intervention plan:"
    )

    # 👇 this line must be aligned with `prompt = (...)`
    output = model(
        prompt,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    generated = output[0]['generated_text'].strip()

    # keep the prompt as part of the final message
    full_text = f"{prompt}\n\n{generated}"

    # generate audio from full text (prompt + AI response)
    audio_bytes = generate_audio(full_text)

    return {
        "prompt": prompt,
        "intervention": generated,
        "full_text": full_text,
        "audio": audio_bytes
    }
# ---------------- Global Logo for Logged-in Pages ----------------
def add_global_logo():
    st.markdown(
        f"""
        <style>
        .top-right-logo {{
            position: fixed;
            top: 15px;
            right: 20px;
            z-index: 9999;
        }}
        .top-right-logo img {{
            max-width: 140px;
        }}
        </style>
        <div class="top-right-logo">
            <img src="file:///C:/Users/Chirag Pradhan/smart india/VITAP mistri coders.png">
        </div>
        """,
        unsafe_allow_html=True
    )
# ---------------- Authentication Pages ----------------
def login_page():
    set_motion_background()
    st.markdown(
        """
        <style>
        .login-card img {
            max-width: 450px;
            margin: auto;
            margin-top: 60px;
            padding: 40px 30px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
            position: center;
        }
        .login-title {
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: #4B0082;
            margin-bottom: 15px;
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
            transition: all 0.2s ease-in-out;
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
    col1, col2 = st.columns([0.4, 0.5])
    with col2:st.image("C:/Users/Chirag Pradhan/smart india/VITAP mistri coders.png", width=120)
    
    st.markdown('<div class="login-title">🔑 Student Dropout Dashboard</div>', unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    # Row for Forgot Password (aligned right)
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("Forgot Password"):
            st.session_state["auth_mode"] = "forgot"
            st.rerun()
    # Login button (centered full width)
    col1, col2 = st.columns([0.4, 0.5])
    with col2:
            if st.button("Login"):
                if username in st.session_state["users"] and st.session_state["users"][username] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    # Don't have an account line
    st.markdown('<p class="center-text">Don’t have an account?</p>', unsafe_allow_html=True)

    # Centered Signup button
    col3, col4, col5 = st.columns([3, 2, 2])
    with col4:
        if st.button("Sign Up"):
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
        .forgot-card img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 15px;
            max-width: 120px;   /* reduced size */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.columns([0.4, 0.5])
    with col2:
        st.image("C:/Users/Chirag Pradhan/smart india/VITAP mistri coders.png", width=120)
    st.markdown('<div class="forgot-title">Forgot Password</div>', unsafe_allow_html=True)
    st.markdown('<div class="forgot-subtitle">Enter your email to reset your password</div>', unsafe_allow_html=True)

    email = st.text_input("Email")
    col1, col2 = st.columns([0.4, 0.5])
    with col2:
        if st.button("Send Reset Link"):
            if email in st.session_state["users"]:
                st.success(f"Password reset link has been sent to {email}")
            else:
                st.error("Email not found in our records")
    col1, col2 = st.columns([0.4, 0.5])
    with col2:
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

            # Ensure required columns exist, fill missing with NaN
            required_cols = st.session_state["students"].columns.tolist()
            for col in required_cols:
                if col not in df_import.columns:
                    df_import[col] = np.nan

            # Clean dtypes (Arrow compatibility)
            df_import = clean_dataframe(df_import)

            # ✅ Ensure student_id exists and is string
            if "registration_number" in df_import.columns:
                df_import["student_id"] = df_import["registration_number"].astype(str).str.strip()
            elif "student_id" in df_import.columns:
                df_import["student_id"] = df_import["student_id"].astype(str).str.strip()
            else:
                st.error("❌ The uploaded file must contain a 'student_id' or 'registration_number' column.")
                return
            st.session_state["students"]["student_id"] = (
                st.session_state["students"]["student_id"].astype(str).str.strip()
            )

            st.subheader("Preview of Imported Data")
            st.dataframe(df_import)   # show full dataset

            if st.button("Import Data"):
                # Drop completely empty rows/cols
                df_import = df_import.dropna(how='all')

                # Keep required columns even if all NaN
                required_cols = st.session_state["students"].columns.tolist()
                for col in required_cols:
                    if col not in df_import.columns:
                        df_import[col] = np.nan

                # Merge with existing data on student_id
                existing = st.session_state["students"]
                # Drop empty rows & columns BEFORE concat, but keep required
                existing = existing.dropna(how="all")
                existing_cols_to_keep = [col for col in existing.columns if col in required_cols or existing[col].notna().any()]
                existing = existing[existing_cols_to_keep]

                df_import_cols_to_keep = [col for col in df_import.columns if col in required_cols or df_import[col].notna().any()]
                df_import = df_import[df_import_cols_to_keep]

                # Now safe concat
                combined = pd.concat([existing, df_import], ignore_index=True)
                combined = combined.drop_duplicates(subset="student_id", keep="last")

                # ✅ keep latest record per student_id
                combined = combined.drop_duplicates(subset="student_id", keep="last")

                # Clean again after merge
                st.session_state["students"] = clean_dataframe(combined)

                st.success(f"✅ Imported {len(df_import)} records. Total students: {len(st.session_state['students'])}")

        except Exception as e:
            st.error(f"Error importing file: {e}")

    st.markdown("### Supported Formats")
    st.markdown("- **CSV**: Comma-separated values")
    st.markdown("- **Excel**: .xlsx or .xls files")
    st.markdown("- **Columns**: The file should include columns matching the student data schema. Missing columns will be filled with defaults.")

# ---------------- Manual Student Entry ----------------
# ---------------- Utility: Clean DataFrame ----------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe is Arrow-compatible by fixing column dtypes"""
    numeric_cols = [
        "attendance", "avg_score", "gpa", "num_attempts",
        "outstanding_dues", "daily_attendance", "subject_attendance",
        "participation", "age"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ensure batch is always a string (avoid mixed types)
    if "batch" in df.columns:
        df["batch"] = df["batch"].astype(str)
    return df
# ---------------- Manual Student Entry ----------------
def manual_input_page():
    set_motion_background()
    st.title("➕ Add Student Data")
    with st.form("student_form"):
        name = st.text_input("Name")
        registration_number = st.text_input("Registration Number")
        age = st.number_input("Age", min_value=18, max_value=100, value=20)
        contact_info = st.text_input("Contact Info")
        degree = st.selectbox("Degree", ["BSc", "BA", "BCom", "MSc", "MA", "PhD"])
        program = st.selectbox("Program", ["BSc CS", "BA Eng", "BCom"])
        semester = st.selectbox("Semester", [1, 2, 3, 4, 5, 6])
        batch = st.text_input("Batch", "2023")
        daily_attendance = st.slider("Daily Attendance %", 0, 100, 80)
        subject_attendance = st.slider("Subject Attendance %", 0, 100, 85)
        attendance_trends = st.text_input("Attendance Trends")
        assessment_scores = st.text_input("Assessment Scores", "Comma separated")
        subject_performance = st.text_input("Subject Performance")
        score = st.slider("Average Score %", 0, 100, 60)
        gpa = st.number_input("GPA", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
        assignment_patterns = st.text_input("Assignment Patterns")
        num_attempts = st.number_input("Number of Attempts", min_value=0, value=1)
        background = st.text_input("Background")
        prev_academic_history = st.text_input("Previous Academic History")
        payment_history = st.text_input("Payment History")
        outstanding_dues = st.number_input("Outstanding Dues", min_value=0.0, value=0.0)
        fee_status = st.selectbox("Fees Status", ["paid", "pending", "overdue"])
        mentor = st.text_input("Mentor ID", "Mentor_A")

        submitted = st.form_submit_button("Save Student")
        if submitted:
            new_row = pd.DataFrame([{
                "name": name,
                "registration_number": registration_number,
                "age": age,
                "contact_info": contact_info,
                "degree": degree,
                "program": program,
                "semester": semester,
                "batch": batch,
                "daily_attendance": daily_attendance,
                "subject_attendance": subject_attendance,
                "attendance_trends": attendance_trends,
                "assessment_scores": assessment_scores,
                "subject_performance": subject_performance,
                "avg_score": score,
                "gpa": gpa,
                "assignment_patterns": assignment_patterns,
                "num_attempts": num_attempts,
                "background": background,
                "prev_academic_history": prev_academic_history,
                "payment_history": payment_history,
                "outstanding_dues": outstanding_dues,
                "fee_status": fee_status,
                "mentor_id": mentor
            }])

            # Exclude all-NA rows and columns to retain old concat behavior
            new_row = new_row.dropna(how='all')
            new_row = new_row.loc[:, new_row.notna().any()]

            # Add the new student to the session
            df = st.session_state["students"]

            # 🔹 Fix ArrowTypeError: ensure consistent datatypes
            if "daily_attendance" in df.columns:
                try:
                    df["daily_attendance"] = pd.to_numeric(df["daily_attendance"], errors="coerce")
                except:
                    df["daily_attendance"] = df["daily_attendance"].astype(str)

            # Optional: auto-fix all object columns
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception:
                        df[col] = df[col].astype(str)


            # ---------------- FIX: Force 'batch' column to string to avoid PyArrow errors ----------------
            # Ensure batch is string
            st.session_state["students"]["batch"] = st.session_state["students"]["batch"].astype(str)

            # Only convert purely numeric columns safely
            for col in ["avg_score", "gpa", "num_attempts", "outstanding_dues"]:
                st.session_state["students"][col] = pd.to_numeric(
                    st.session_state["students"][col], errors="coerce"
                )
            st.success(f"Student {name} added!")
            # Append the new row to the students dataframe in session state
            st.session_state["students"] = pd.concat([st.session_state["students"], new_row], ignore_index=True)
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

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Ensure 'name' column exists
    if "name" not in df.columns:
        df["name"] = df["registration_number"]  # fallback

    if df.empty: 
        st.warning("No student data available yet. Add some manually!") 
        return

    # Summary KPIs
    total_students = len(df)
    dropout_risk = df[df["risk_level"] == "High"].shape[0]
    avg_attendance = df["daily_attendance"].mean()
    avg_score = df["avg_score"].mean()
    avg_gpa = df["gpa"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Students", total_students)
    col2.metric("High Risk Dropouts", dropout_risk)
    col3.metric("Avg Attendance", f"{avg_attendance:.2f}%")
    col4.metric("Avg Score", f"{avg_score:.2f}")
    col5.metric("Avg GPA", f"{avg_gpa:.2f}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "👨‍🎓 Students", "🚨 Alerts",
        "⚙️ Threshold Preview", "🛠 Additional Notes"
    ])

    # ---------------- Tab 1: Overview ----------------
    with tab1:
        st.subheader("📌 Attendance vs Scores")
        fig_scatter = px.scatter(
            df,
            x="daily_attendance",
            y="avg_score",
            color="risk_level",
            size="gpa",
            hover_data=["name", "registration_number", "gpa", "mentor_id"]
        )
        st.plotly_chart(fig_scatter)

        st.subheader("Fee Status Distribution")
        if 'fee_status' in df.columns:
            fee_counts = df["fee_status"].value_counts().reset_index()
            fee_counts.columns = ["fee_status", "count"]
            fig2 = px.bar(fee_counts, x="fee_status", y="count", title="Fee Status Distribution")
            st.plotly_chart(fig2)
        else:
            st.warning("No 'fee_status' column available in the data.")

        st.subheader("Attendance Trend Example")
        sample = df.sample(1).iloc[0]
        trend = pd.Series(
            np.clip(
                np.random.normal(sample["daily_attendance"], 5, 8),
                30, 100
            ),
            index=pd.date_range(datetime.today()-timedelta(weeks=7), periods=8, freq="W")
        )
        st.line_chart(trend)

        st.subheader("📊 Cohort Radar Chart")
        df["daily_attendance"] = pd.to_numeric(df["daily_attendance"], errors="coerce")
        df["avg_score"] = pd.to_numeric(df["avg_score"], errors="coerce")
        df["gpa"] = pd.to_numeric(df["gpa"], errors="coerce")

        metrics = {
            "Avg Attendance": df["daily_attendance"].mean(skipna=True),
            "Avg Score": df["avg_score"].mean(skipna=True),
            "Avg GPA": df["gpa"].mean(skipna=True),
            "High Risk %": (df["risk_level"]=="High").mean(skipna=True) * 100,
            "Fee Defaulters %": (df["fee_status"]!="paid").mean(skipna=True) * 100 if 'fee_status' in df.columns else 0,
        }

        radar_df = pd.DataFrame(dict(
            r=list(metrics.values()),
            theta=list(metrics.keys())
        ))
        fig = px.line_polar(radar_df, r="r", theta="theta", line_close=True)
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Degree Distribution")
        degree_counts = df["degree"].value_counts().reset_index()
        degree_counts.columns = ["degree", "count"]
        fig3 = px.pie(degree_counts, names="degree", values="count", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("GPA Distribution")
        fig5 = px.histogram(df, x="gpa", nbins=10, color_discrete_sequence=["#636efa"])
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Age Distribution")
        fig6 = px.histogram(df, x="age", nbins=10, color_discrete_sequence=["#ef553b"])
        st.plotly_chart(fig6, use_container_width=True)

    # ---------------- Tab 2: Students ----------------
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
        sid = st.selectbox("Select Student", df["student_id"] if not df.empty else [])

        if not df.empty and sid in df["student_id"].values:
            student = df[df["student_id"] == sid].iloc[0]

            st.subheader(f"📋 Details for {student.get('name', 'Unknown Student')} (ID: {sid})")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Attendance %", student.get("daily_attendance", "N/A"))
            col2.metric("GPA", student.get("gpa", "N/A"))
            col3.metric("Attempts", student.get("num_attempts", "N/A"))
            col4.metric("Fees Status", student.get("fee_status", "N/A"))
        else:
            st.warning("⚠️ No student data available. Please add or import students first.")

        st.subheader("🗑️ Delete Students")
        if not df.empty:
            students_to_delete = st.multiselect("Select students to delete", df["student_id"].tolist())
            if st.button("Delete Selected Students"):
                if students_to_delete:
                    st.session_state["students"] = st.session_state["students"][~st.session_state["students"]["student_id"].isin(students_to_delete)]
                    st.success(f"Deleted {len(students_to_delete)} student(s).")
                    st.rerun()
                else:
                    st.warning("No students selected for deletion.")
        else:
            st.info("No students to delete.")

        st.subheader("Mentor Notes")
        if not df.empty and sid in df["student_id"].values:
            student = df[df["student_id"] == sid].iloc[0]

            if st.button("Generate Mentor's Consulted Intervention Plan", key=f"ai_{sid}"):
                with st.spinner("Generating plan..."):
                    ai_notes = generate_intervention(student)
                    st.text_area("AI Suggested Notes", value=ai_notes["intervention"], height=150)

                    try:
                        st.audio(ai_notes["audio"], format="audio/wav")
                    except Exception as e:
                        st.warning(f"Audio generation failed: {e}. TTS may not be available on this system.")
            else:
                st.text_area("AI Suggested Notes", placeholder="Click the button to generate an intervention plan...")
        else:
            st.info("👆 Please select a student above to generate mentor notes.")

    # ---------------- Tab 3: High Risk Alerts ----------------
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

    # ---------------- Tab 4: Threshold Preview ----------------
    with tab4:
        st.subheader("⚙️ Threshold Live Preview")
        att_thresh = st.slider("Min Attendance %", 0, 100, 60)
        score_thresh = st.slider("Min Avg Score %", 0, 100, 50)

        preview_df = compute_risk(st.session_state["students"], att_thresh, score_thresh)
        st.dataframe(preview_df[["student_id","name","daily_attendance","avg_score","gpa","risk_level"]])

    # ---------------- Tab 5: Developer Notes ----------------
    with tab5:
        st.subheader("✨ Fun Dashboard Facts")
        st.markdown(f"""
        - 🎓 **Total Students:** {total_students} in the system.  
        - 🚨 **High Risk Cases:** {dropout_risk} flagged for mentor review.  
        - 📈 **Average Attendance:** {avg_attendance:.2f}% across the cohort.  
        - 🧮 **Average Score:** {avg_score:.2f}.  
        - 🌟 **Average GPA:** {avg_gpa:.2f}.  
        - 💡 Keep exploring — this dashboard evolves as you add more data!
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