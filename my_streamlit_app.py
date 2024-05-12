import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Classifying sentiments ")
text_input = st.text_input("Please enter text:")

if text_input:
    text_vectorized = vectorizer.transform([text_input])
    prediction = model.predict(text_vectorized)

    if prediction == 1:
        st.write("Positive: إيجابي")
    elif prediction == -1:
        st.write("Negative: سلبي")
    else:
        st.write("Neutral: محايد")
