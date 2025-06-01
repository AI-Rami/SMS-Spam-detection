import streamlit as st
import joblib
import re

# title 
st.set_page_config(page_title="SMS Spam Detection", layout="centered")
st.title(" SMS Spam Detection App")
st.write("Enter your message and choose a model to classify it as **Spam** or **Ham**.")

# loading the models and vectorizer 
@st.cache_resource
def load_models():
    logistic_model = joblib.load("logistic_regression_tfidf_model.pkl")
    svm_model = joblib.load("svm_tfidf_model.pkl")
    nb_model = joblib.load("multinomialnb_tfidf_model.pkl")  
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return logistic_model, svm_model, nb_model, vectorizer

logistic_model, svm_model, nb_model, vectorizer = load_models()

model_options = {"Logistic Regression (TF-IDF)": logistic_model,"SVM (TF-IDF)": svm_model,"Naive Bayes (MultinomialNB, TF-IDF)": nb_model}

# the cleaning function 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Input  
message = st.text_area("Enter your message:")

# to be able to select model 
model_choice = st.selectbox(" Choose a model:", list(model_options.keys()))

# Predict 
if st.button(" Predict"):
    if message:
        cleaned = clean_text(message)
        message_vec = vectorizer.transform([cleaned])
        model = model_options[model_choice]
        prediction = model.predict(message_vec)[0]
        label = "Spam" if prediction == 1 else "Ham"
        
        st.success(f"**Prediction:** {label}")
        st.write(" Message Shape:", message_vec.shape)
        st.write(" Non-zero Elements:", message_vec.nnz)
    else:
        st.warning(" Please enter a message.")
