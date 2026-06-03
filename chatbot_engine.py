from groq import Groq

# ---------------- GROQ CLIENT ----------------

client = Groq(

    api_key="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"
)

# ---------------- CHATBOT FUNCTION ----------------

def generate_chat_response(

    user_message,

    emotion_context="Unknown"
):

    chatbot_prompt = f"""

    You are HealthcareGPT,
    a professional AI healthcare assistant.

    Your responsibilities:

    - emotional wellness support
    - stress management guidance
    - healthy lifestyle suggestions
    - productivity advice
    - motivational support

    Never provide dangerous medical advice.

    Always recommend consulting doctors for serious conditions.

    Current emotional context of user:
    {emotion_context}

    User message:
    {user_message}
    """

    # ---------------- GROQ RESPONSE ----------------

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {
                "role": "system",

                "content":
                "You are a professional AI healthcare assistant."
            },

            {
                "role": "user",

                "content": chatbot_prompt
            }
        ]
    )

    bot_reply = response.choices[0].message.content

    return bot_reply