import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./best_bert_model')  # Path where your model is saved
tokenizer = BertTokenizer.from_pretrained('./best_bert_model')  # Path where your tokenizer is saved

# Ensure the model is in evaluation mode
model.eval()

# Function to predict sentiment (for example)
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # Don't track gradients during inference
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# Define class labels (based on your training)
class_labels = ['Negative', 'Neutral', 'Positive']  # Modify this if you have more classes

# Streamlit UI
st.title("🧑‍💻 Review Sentiment Classifier")

st.write(
    "Enter a product review below, and the model will predict whether the sentiment is **Negative**, **Neutral**, or **Positive**."
)

# Get user input for review text
user_input = st.text_area("Enter Product Review:", "")

if st.button("Classify"):
    if user_input.strip():
        # Get the prediction
        predicted_class_idx = predict_sentiment(user_input)
        
        # Convert index to label
        sentiment_label = class_labels[predicted_class_idx]
        
        # Display the results
        st.write(f"### **Prediction: {sentiment_label}**")
    else:
        st.warning("Please enter some text.")



