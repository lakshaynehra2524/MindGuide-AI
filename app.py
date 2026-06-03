import streamlit as st
import pandas as pd
import numpy as np
import joblib

from database import (
    create_user,
    save_report,
    get_user_reports
)

from chatbot.chatbot_engine import generate_chat_response

# ---------------- PAGE CONFIG ----------------

st.set_page_config(

    page_title="HealthcareGPT",

    page_icon="🩺",

    layout="centered"
)
# ---------------- CUSTOM UI DESIGN ----------------

st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Main Background */

.stApp {

    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #0f172a
    );

    color: white;
}

/* Main Title */

h1 {

    color: #38bdf8 !important;

    text-align: center;

    font-size: 3rem !important;

    font-weight: 700 !important;
}

/* Subheaders */

h2, h3 {

    color: #f8fafc !important;
}

/* Input Boxes */

.stTextInput input,
.stTextArea textarea,
.stSelectbox div,
.stNumberInput input {

    background-color: rgba(255,255,255,0.08) !important;

    color: white !important;

    border-radius: 12px !important;

    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* Buttons */

.stButton button {

    background: linear-gradient(
        90deg,
        #38bdf8,
        #0ea5e9
    );

    color: white;

    border: none;

    border-radius: 12px;

    padding: 12px 25px;

    font-weight: bold;

    transition: 0.3s;
}

.stButton button:hover {

    transform: scale(1.03);

    background: linear-gradient(
        90deg,
        #0ea5e9,
        #38bdf8
    );
}

/* Chat Messages */

[data-testid="stChatMessage"] {

    background-color: rgba(255,255,255,0.05);

    border-radius: 15px;

    padding: 10px;

    margin-bottom: 10px;

    border: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar */

section[data-testid="stSidebar"] {

    background-color: #111827;
}

/* Success Box */

.stSuccess {

    border-radius: 12px;
}

/* Info Box */

.stInfo {

    border-radius: 12px;
}

</style>

""", unsafe_allow_html=True)

# --------------- LOAD MODELS ---------------

vectorizer = joblib.load("models/vectorizer.pkl")
le_ambience_type = joblib.load("models/le_ambience_type.pkl")
le_time_of_day = joblib.load("models/le_time_of_day.pkl")
le_previous_day_mood = joblib.load("models/le_previous_day_mood.pkl")
le_reflection_quality = joblib.load("models/le_reflection_quality.pkl")
le_target_em = joblib.load("models/le_emotional_state.pkl")
le_stress_level = joblib.load("models/le_stress_level.pkl")

model = joblib.load(
    "models/emotion_model.pkl"
)

# ---------------- TITLE ----------------

st.title("🩺 HealthcareGPT")

st.subheader(
    "AI-Powered Mental Wellness Assistant"
)

st.info(
    "This system provides emotional wellness support and is not a replacement for professional medical advice."
)

# ---------------- SESSION STATE ----------------

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- USER REGISTRATION ----------------

st.header("👤 User Registration")

name = st.text_input("Enter Your Name")

age = st.number_input(
    "Enter Your Age",
    min_value=1,
    max_value=100
)

gender = st.selectbox(
    "Select Gender",
    ["Male", "Female", "Other"]
)

if st.button("Create User Profile"):

    user_id = create_user(
        name,
        age,
        gender
    )

    st.session_state.user_id = user_id

    st.success(
        f"User profile created successfully. Your User ID is {user_id}"
    )

# ---------------- MAIN SYSTEM ----------------

if st.session_state.user_id:

    st.header("🧠 Emotional Analysis")

    journal_text = st.text_area(
        "Describe how you are feeling today"
    )

    ambience_type = st.selectbox(
        "Ambience Type",
        [
            "quiet",
            "office",
            "home",
            "crowded",
            "nature",
            "library"
        ]
    )

    duration_min = st.slider(
        "Activity Duration (minutes)",
        10,
        180,
        60
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        1.0,
        12.0,
        7.0
    )

    energy_level = st.slider(
        "Energy Level",
        1,
        10,
        5
    )

    stress_level = st.selectbox(
        "Stress Level",
        ["Low", "Moderate", "High"]
    )

    time_of_day = st.selectbox(
        "Time of Day",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Night"
        ]
    )

    previous_day_mood = st.selectbox(
        "Previous Day Mood",
        [
            "happy",
            "sad",
            "tired",
            "motivated",
            "anxious",
            "normal"
        ]
    )

    reflection_quality = st.selectbox(
        "Reflection Quality",
        [
            "low",
            "medium",
            "high"
        ]
    )

    # ---------------- PREDICT BUTTON ----------------

    if st.button("Analyze Emotional State"):

         df = pd.DataFrame([{

            "journal_text": journal_text,

            "ambience_type": ambience_type,

            "duration_min": duration_min,

            "sleep_hours": sleep_hours,

            "energy_level": energy_level,

            "stress_level": stress_level,

            "time_of_day": time_of_day,

            "previous_day_mood": previous_day_mood,

            "reflection_quality": reflection_quality
        }])

         text_vector = vectorizer.transform(df["journal_text"]).toarray()
         # Transforms the  journal_text same according to the backend text transformation
         df['ambience_type'] = le_ambience_type.transform(df['ambience_type'])
         df['time_of_day'] = le_time_of_day.transform(df['time_of_day'])
         df['previous_day_mood'] = le_previous_day_mood.transform(df['previous_day_mood'])
         df['reflection_quality'] = le_reflection_quality.transform(df['reflection_quality'])
         df['stress_level'] = le_stress_level.transform(df['stress_level'])

         # Tranforms the categorical columns into numerical using the backend labelencoder 
         numerical_data = df.drop(["journal_text"] , axis = 1)
         # All numerical data 
         final_input = np.concatenate([text_vector, numerical_data], axis=1)
        # final input to the model by concatinating all processed columns 

        # ---------------- PREDICTION ----------------

         prediction_encoded = model.predict(final_input)[0]

         prediction = le_target_em.inverse_transform(
             [prediction_encoded]
         )[0]

        # ---------------- RECOMMENDATIONS ----------------

         recommendations = {

            "calm":
            "Maintain your healthy routine and continue mindfulness practices.",

            "focused":
            "Excellent productivity state. Maintain balance and hydration.",

            "restless":
            "Try breathing exercises and reduce overthinking triggers.",

            "overwhelmed":
            "Take adequate rest and reduce workload. Consider meditation.",

            "neutral":
            "Maintain a balanced lifestyle and healthy sleep cycle.",

            "mixed":
            "Try journaling and structured relaxation techniques."
        }

         recommendation = recommendations.get(
            prediction,
            "Maintain healthy habits."
        )

        # ---------------- DISPLAY RESULTS ----------------

         st.success(
            f"Predicted Emotional State: {prediction}"
        )

         st.info(
            f"Healthcare Recommendation: {recommendation}"
        )

        # ---------------- SAVE REPORT ----------------

         save_report(

            st.session_state.user_id,

            prediction,

            stress_level,

            recommendation,

            journal_text
        )

         st.success(
            "Report saved successfully."
        )

    # ---------------- PREVIOUS REPORTS ----------------

    st.header("📋 Previous Reports")

    reports = get_user_reports(
        st.session_state.user_id
    )

    if reports:

        for report in reports:

            st.write("---")

            st.write(
                f"Emotion: {report[2]}"
            )

            st.write(
                f"Stress Level: {report[3]}"
            )

            st.write(
                f"Recommendation: {report[4]}"
            )

            st.write(
                f"Journal: {report[5]}"
            )

            st.write(
                f"Date: {report[6]}"
            )


# ---------------- AI CHATBOT ----------------

st.header("💬 HealthcareGPT Assistant")

user_message = st.chat_input(
    "Ask your healthcare question..."
)

if user_message:

    # SAVE USER MESSAGE

    st.session_state.chat_history.append(
        ("user", user_message)
    )

    # DEFAULT EMOTION

    current_emotion = "Unknown"

    # USE MODEL PREDICTION IF AVAILABLE

    if 'prediction' in locals():
        current_emotion = prediction

    # AI RESPONSE

    bot_reply = generate_chat_response(

        user_message,

        current_emotion
    )

    # SAVE AI RESPONSE

    st.session_state.chat_history.append(
        ("assistant", bot_reply)
    )

# ---------------- DISPLAY CHAT ----------------

for role, message in st.session_state.chat_history:

    with st.chat_message(role):

        st.markdown(message)