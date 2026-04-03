# Importing libraries 
import numpy as np 
import pandas as pd 
import streamlit as st 
import joblib 

# Adding home title and Discription 
st.title("MindGuide - AI")
st.write("This is my AI model which predicts the stress levels and emotional state of the user")

# Loading models from the backend 
em_model = joblib.load("em_model.pkl")
st_model = joblib.load("st_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le_ambience_type = joblib.load("le_ambience_type.pkl")
le_time_of_day = joblib.load("le_time_of_day.pkl")
le_previous_day_mood = joblib.load("le_previous_day_mood.pkl")
le_face_emotion_hint = joblib.load("le_face_emotion_hint.pkl")
le_reflection_quality = joblib.load("le_reflection_quality.pkl")
le_target_em = joblib.load("le_emotional_state.pkl")
le_target_st = joblib.load("le_stress_level.pkl")

# Adding input sections :
journal_text = st.text_area("Enter your Journal Text here .....")
ambience_type = st.radio("Choose Ambience type :" , ["ocean", "mountain", "forest", "cafe", "rain"])
duration_min = st.number_input("Enter time spent in the Emotion in minutes :" , min_value = 1 , max_value = 240)
sleep_hours = st.number_input("Enter time spent sleeping in hours :" , min_value = 0 , max_value = 18)
energy_level = st.number_input("Enter energy level on scale of 1 to 5 :" , min_value = 1 , max_value = 5)
time_of_day = st.radio("Choose time of day :" , ["early_morning", "morning", "afternoon", "evening", "night"])
previous_day_mood = st.radio("Choose mood type of previous day :" , 
                             ["neutral","calm", "focused", "overwhelmed", "mixed", "restless"])
face_emotion_hint = st.radio("Choose face type emotion :" , 
                             ["neutral_face","calm_face", "happy_face", "tired_face", "tense_face", "none"])
reflection_quality = st.radio("Choose more clear vision of your thoughts :" , 
                             ["clear","vague", "conflicted"])
intensity = st.number_input("Enter intensity of emotions on scale of 1 to 5 :" , min_value = 1 , max_value = 5)

df = pd.DataFrame([{
    "journal_text": journal_text,
    "ambience_type": ambience_type,
    "duration_min": duration_min,
    "sleep_hours": sleep_hours,
    "energy_level": energy_level,
    "time_of_day": time_of_day,
    "previous_day_mood": previous_day_mood,
    "face_emotion_hint": face_emotion_hint,
    "reflection_quality": reflection_quality,
    "intensity": intensity
}]) 
if st.button("Submit"):
    # Data transformation as required in input
    text_vector = vectorizer.transform(df["journal_text"]).toarray()
    # Transforms the  journal_text same according to the backend text transformation
    df['ambience_type'] = le_ambience_type.transform(df['ambience_type'])
    df['time_of_day'] = le_time_of_day.transform(df['time_of_day'])
    df['previous_day_mood'] = le_previous_day_mood.transform(df['previous_day_mood'])
    df['face_emotion_hint'] = le_face_emotion_hint.transform(df['face_emotion_hint'])
    df['reflection_quality'] = le_reflection_quality.transform(df['reflection_quality'])
    # Tranforms the categorical columns into numerical using the backend labelencoder 
    numerical_data = df.drop(columns=["journal_text"]).values
    # All numerical data 
    final_input = np.concatenate([text_vector, numerical_data], axis=1)
    # final input to the model by concatinating all processed columns 
    prediction_em = em_model.predict(final_input)
    prediction_st = st_model.predict(final_input)
    decoded_prediction_em = le_target_em.inverse_transform(prediction_em)
    decoded_prediction_st = le_target_st.inverse_transform(prediction_st)
    st.success("Prediction completed!")
    st.write("Predicted Emotion:", decoded_prediction_em[0])
    # Confidence score for emotional state 
    probab_em = em_model.predict_proba(final_input)
    pred_index_em = prediction_em[0]
    # Confidence score
    confidence_em = probab_em[0][pred_index_em]
    st.progress(float(confidence_em))
    st.write("Emotion Confidence:", round(confidence_em * 100, 2), "%")

    st.write("Predicted Stress Level:", decoded_prediction_st[0])
    # Confidence score for stress level 
    proba_st = st_model.predict_proba(final_input)
    pred_index_st = prediction_st[0]
    # Confidence score 
    confidence_st = proba_st[0][pred_index_st]
    st.progress(float(confidence_st))
    st.write("Stress Confidence:", round(confidence_st * 100, 2), "%")
        
    
    recommendations = {
        "focused_low": [
            "You're in a great flow state, keep going",
            "Use this time for deep work or learning"],
            
        "focused_medium": [
            "Stay consistent but take short breaks",
            "Avoid burnout by pacing yourself"
        ],
        "focused_high": [
            "You may be overworking, pause briefly",
            "Try breathing exercises to stay balanced"
        ],

        "calm_low": [
            "Maintain this peaceful state",
            "Good time for reflection or light activities"
        ],
        "calm_medium": [
            "Stay mindful and avoid distractions",
            "Try meditation to deepen calmness"
        ],
        "calm_high": [
            "Something may be building underneath",
            "Pause and check in with your thoughts"
        ],

        "neutral_low": [
            "Try engaging in something creative",
            "A good time to plan your day"
        ],
        "neutral_medium": [
            "Shift into a productive task",
            "Small actions can improve your mood"
        ],
        "neutral_high": [
            "You might be suppressing stress",
            "Take a break and reset your focus"
        ],

        "restless_low": [
            "Channel energy into physical activity",
            "Try a quick walk or stretching"
        ],
        "restless_medium": [
            "Break tasks into smaller steps",
            "Avoid multitasking"
        ],
        "restless_high": [
            "Pause everything for a moment",
            "Practice deep breathing or grounding"
        ],

        "overwhelmed_low": [
            "Start with one small task",
            "Keep things simple and manageable"
        ],
        "overwhelmed_medium": [
            "Prioritize tasks and remove distractions",
            "Talk to someone if needed"
        ],
        "overwhelmed_high": [
            "Stop and take a proper break",
            "Focus only on essentials right now",
            "Consider reaching out for support"]}

    emotion = decoded_prediction_em[0]
    stress = decoded_prediction_st[0]

    key = f"{emotion}_{stress}"
    if key in recommendations:
        st.subheader("💡 Suggestions for you:")
        for rec in recommendations[key]:
            st.write("~", rec)

# Background 
def set_bg():
    # Background
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1743401853978-d033df48fab6?q=80&w=1332&auto=format&fit=crop");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: -1;
        }
        </style>
            """,
        unsafe_allow_html=True
    )

set_bg()