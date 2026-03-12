# dropout_dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

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
        "attendance", "avg_score", "fees_status", "mentor_id"
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

# ---------------- Authentication Pages ----------------
def login_page():
    st.title("🔑 Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["logged_in"] = True
            st.session_state["current_user"] = username
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.markdown("---")
    if st.button("Sign Up"):
        st.session_state["auth_mode"] = "signup"
        st.rerun()
    if st.button("Forgot Password"):
        st.session_state["auth_mode"] = "forgot"
        st.rerun()

def signup_page():
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

def forgot_password_page():
    st.title("🔒 Forgot Password")

    username = st.text_input("Enter your username")
    if st.button("Reset Password"):
        if username in st.session_state["users"]:
            st.info(f"A reset link has been sent to {username}@example.com (simulated).")
        else:
            st.error("Username not found.")

    if st.button("Back to Login"):
        st.session_state["auth_mode"] = "login"
        st.rerun()

# ---------------- Manual Student Entry ----------------
def manual_input_page():
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
                [st.session_state["students"], new_row],
                ignore_index=True
            )
            st.success(f"Student {name} added!")

    if not st.session_state["students"].empty:
        st.subheader("📋 Current Student Records")
        st.dataframe(st.session_state["students"])

# ---------------- Dashboard ----------------
def dashboard_page():
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
        st.bar_chart(df["risk_level"].value_counts())

        st.subheader("Fee Status Distribution")
        st.bar_chart(df["fees_status"].value_counts())

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

        # Radar Chart
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

        # CSV Download for All Students
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

        st.text_area("Mentor Notes", placeholder="Enter intervention plan here...")

    with tab3:
        st.subheader("High Risk Students")
        risky = df[df["risk_level"]=="High"]
        if risky.empty:
            st.success("No students at high risk 🎉")
        else:
            st.dataframe(risky)

            # CSV Download for High Risk Students
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
        """)

# ---------------- Main Router ----------------
def main():
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    if not st.session_state["logged_in"]:
        if st.session_state["auth_mode"] == "login":
            login_page()
        elif st.session_state["auth_mode"] == "signup":
            signup_page()
        elif st.session_state["auth_mode"] == "forgot":
            forgot_password_page()
    else:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Dashboard", "Add Student", "Logout"])
        if page == "Dashboard":
            dashboard_page()
        elif page == "Add Student":
            manual_input_page()
        elif page == "Logout":
            st.session_state["logged_in"] = False
            st.session_state["current_user"] = None
            st.session_state["auth_mode"] = "login"
            st.rerun()

if __name__ == "__main__":
    main()
