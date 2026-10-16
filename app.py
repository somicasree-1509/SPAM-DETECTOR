"""
app.py
Streamlit web app for the Spam Message Detector.

Run with:  streamlit run app.py
"""

import os
import joblib
import streamlit as st

MODEL_FILE = "spam_model.joblib"

st.set_page_config(page_title="Spam Message Detector", page_icon="📩")


@st.cache_resource
def load_model():
    """Load the trained model once. Returns None if it doesn't exist yet."""
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


st.title("📩 Spam Message Detector")
st.write("Type or paste a message below and the app will tell you if it looks like spam.")

model = load_model()

if model is None:
    st.error(
        "Model file not found. Please train the model first by running "
        "`python train_model.py` in the terminal, then refresh this page."
    )
    st.stop()

message = st.text_area("Enter your message:", height=150, placeholder="e.g. You won a free prize! Click here...")

if st.button("Check Message"):
    if not message.strip():
        st.warning("Please enter a message first. 🙂")
    else:
        prediction = model.predict([message])[0]
        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

        # Show how confident the model is
        spam_probability = model.predict_proba([message])[0][1]
        st.caption(f"Spam probability: {spam_probability:.0%}")
