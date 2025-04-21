import os
os.system('pip install torch')
import gdown
import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Constants ---
MODEL_PATH = "best_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200
GDRIVE_MODEL_ID = "1y8_ss47dlFzLCql3hXdAZ_XfnK-hfUsl"

# --- Download model if not exists ---
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}", MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

    
# Load trained model
model = tf.keras.models.load_model("best_model.h5")

# Load tokenizer
tokenizer = joblib.load("tokenizer.pkl")

# Constants (ensure this matches what you used in training)
MAX_LEN = 200  # Length used in pad_sequences

# Streamlit UI
st.title("📰 Fake News Detector")

st.write("Enter a news article below, and the model will predict whether it's **Real** or **Fake**.")

user_input = st.text_area("Enter News Article:", "")

if st.button("Classify"):
    if user_input.strip():
        # Preprocess input text
        input_seq = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_seq, maxlen=MAX_LEN, padding="post", truncating="post")
        
        # Make prediction
        prediction = model.predict(input_padded)[0][0]  # Assuming binary classification (0 = Fake, 1 = Real)
        
        # Determine label
        label = "🟢 Real News" if prediction >= 0.5 else "🔴 Fake News"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        
        # ✅ Fixed this line (properly closed f-string)
        st.write(f"### **Prediction: {label}**")
        st.write(f"🧠 Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter some text.")
