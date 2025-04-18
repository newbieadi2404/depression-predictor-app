


import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# Load trained components
model = joblib.load("student_depression_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("feature_columns.pkl")

PREDICTION_LOG = "predictions_log.csv"

def main():
    st.set_page_config(page_title="Student Depression Risk", page_icon="🧠", layout="wide")


    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3875/3875144.png", width=100)
        st.title("🧠 Mental Health Prediction")
        st.markdown("---")
        st.markdown("Welcome! This tool predicts a student's risk of depression using lifestyle and academic factors.")
        st.markdown("Made with ❤️ by *Aditya Mishra*")
        st.markdown("---")
        st.markdown("📧 Contact: chintu01032005@gmail.com")

    # Main Title
    st.markdown("""
        <div style="background-color: #f9f9f9; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h1 style="color: #333;">🧠 Student Depression Prediction</h1>
            <p style="color: gray;">Use this tool to estimate depression risk based on inputs.</p>
        </div>
    """, unsafe_allow_html=True)

    # Clear state
    if "clear" not in st.session_state:
        st.session_state.clear = False

    # Inputs
    st.markdown("## 📝 Student Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Age", min_value=10, max_value=100, value=20)
        academic_pressure = st.slider("📚 Academic Pressure", 0, 10, 5)
        work_pressure = st.slider("💼 Work Pressure", 0, 10, 5)
        cgpa = st.number_input("📊 CGPA", min_value=0.0, max_value=10.0, value=7.0, format="%.2f")
        study_satisfaction = st.slider("📖 Study Satisfaction", 0, 10, 5)
        job_satisfaction = st.slider("🧳 Job Satisfaction", 0, 10, 5)

    with col2:
        sleep_duration = st.number_input("😴 Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0, format="%.2f")
        work_study_hours = st.number_input("⏱️ Work/Study Hours", min_value=0, max_value=24, value=8)
        city = st.selectbox("🏙️ City", ["Urban", "Suburban", "Rural"])
        degree = st.selectbox("🎓 Degree Program", ["Engineering", "Arts", "Science", "Business"])
        dietary_habits = st.selectbox("🥗 Dietary Habits", ["Healthy", "Unhealthy", "Moderate"])

    st.markdown("## 🧠 Mental & Emotional Factors")

    col3, col4 = st.columns(2)
    with col3:
        family_history = st.selectbox("🧬 Family History of Mental Illness", ["Yes", "No"])
        financial_stress = st.slider("💸 Financial Stress", 0, 10, 5)

    with col4:
        relationship_issues = st.selectbox("💔 Relationship Issues", ["Yes", "No"])
        support_system = st.selectbox("👨‍👩‍👧 Support System Available", ["Yes", "No"])
        substance_use = st.selectbox("🚬 Substance Use", ["Yes", "No"])

    total_pressure = academic_pressure + work_pressure

    st.markdown("## 🚀 Predict")

    col5, col6 = st.columns([1, 1])
    with col5:
        predict_clicked = st.button("🔍 Predict Risk")
    with col6:
        if st.button("🔄 Clear All"):
            st.rerun()

    if predict_clicked:
        input_data = {
            "Age": age,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "CGPA": cgpa,
            "Study Satisfaction": study_satisfaction,
            "Job Satisfaction": job_satisfaction,
            "Sleep Duration": sleep_duration,
            "Work/Study Hours": work_study_hours,
            "Total Pressure": total_pressure,
            "City": city,
            "Degree": degree,
            "Dietary Habits": dietary_habits,
            "Family History of Mental Illness": family_history,
            "Financial Stress": financial_stress,
            "Relationship Issues": relationship_issues,
            "Support System Available": support_system,
            "Substance Use": substance_use
        }

        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df)

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]

        try:
            scaled_input = scaler.transform(df)
            prediction = model.predict(scaled_input)[0]
            prob = model.predict_proba(scaled_input)[0][1]

            result = "🟡 At Risk of Depression" if prediction == 1 else "🟢 No Depression Risk"

            st.markdown(f"""
    <div style="background-color: #e8f0fe; padding: 15px; border-left: 6px solid #1a73e8; border-radius: 10px;">
        <h4 style="color: #1a1a1a; margin-bottom: 10px;">Prediction Result:</h4>
        <p style="font-size: 18px; color: #000000; margin-top: 0;">{result}</p>
        <p style="color: #000000;"><b>📈 Risk Score:</b> {prob * 100:.1f}%</p>
    </div>
""", unsafe_allow_html=True)


            st.progress(prob)

            # Save prediction
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {
                "Timestamp": timestamp,
                "Prediction": prediction,
                "Risk Score": round(prob, 4),
                **input_data
            }

            log_df = pd.DataFrame([log_entry])
            if os.path.exists(PREDICTION_LOG):
                log_df.to_csv(PREDICTION_LOG, mode='a', header=False, index=False)
            else:
                log_df.to_csv(PREDICTION_LOG, index=False)

            # Download button
            if os.path.exists(PREDICTION_LOG):
                with open(PREDICTION_LOG, "rb") as file:
                    st.download_button("⬇️ Download Predictions (CSV)", file, "predictions_log.csv", "text/csv")

            # Show history
            st.markdown("## 📜 Recent Predictions")
            log_data = pd.read_csv(PREDICTION_LOG)
            st.dataframe(log_data.tail(10))

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Path to your predictions log file
PREDICTION_LOG = "predictions_log.csv"

st.title("📊 Recent Depression Predictions Overview")

# Check if log file exists
if os.path.exists(PREDICTION_LOG):
    df = pd.read_csv(PREDICTION_LOG)

    if df.empty:
        st.warning("No predictions have been logged yet.")
    else:
        # Get last 10 predictions
        recent = df.tail(10)

        # Convert Prediction to readable labels
        recent["Label"] = recent["Prediction"].map({1: "At Risk", 0: "No Risk"})

        # Plot bar chart
        fig = px.bar(
            recent,
            x="Timestamp",
            y="Risk Score",
            color="Prediction",
            text="Label",
            color_discrete_map={1: "crimson", 0: "green"},
            title="Risk Scores from Recent Predictions",
            labels={"Risk Score": "Depression Risk"},
        )
        fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)

        # Optional: show raw data below
        #with st.expander("🔍 View Raw Data"):
            #st.dataframe(recent)

else:
    st.error("Prediction log not found. Please run some predictions first.")
