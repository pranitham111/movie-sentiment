import streamlit as st
import pickle as pk
from nltk.corpus import stopwords
import nltk
from streamlit_lottie import st_lottie
import json
import requests

nltk.download('stopwords')

# Load model and vectorizer
model = pk.load(open('model.pkl', 'rb'))
vectorizer = pk.load(open('vectorizer.pkl', 'rb'))

# Load Lottie animations
def load_lottie(path):
    with open(path, 'r') as f:
        return json.load(f)

lottie_positive = load_lottie("positive.json")
lottie_negative = load_lottie("negative.json")

# Clean review text
def clean_review(review):
    return ' '.join(word for word in review.split() if word not in stopwords.words('english'))

# Setup Streamlit page config
st.set_page_config(page_title="Movie Review Sentiment", layout="centered")

# Session state to manage pages
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Home Page
if st.session_state.page == 'home':
    st.title("🎬 Movie Review Sentiment Analysis")
    st.markdown("""
        Welcome to our Machine Learning project!  
        This app predicts whether a movie review is **Positive** or **Negative** using a Logistic Regression model trained on IMDB dataset.
        
        **Team Members**:
        - Malla Pranitham
        - Sinka Ramu 
        - Velichetti Hari Charan
    """)
    st.markdown("---")
    if st.button("🚀 Start"):
        st.session_state.page = 'predict'
        st.rerun()


# Prediction Page
elif st.session_state.page == 'predict':
    st.title("🔍 Sentiment Predictor")

    review = st.text_input("Enter your movie review:")
    
    if st.button("Predict"):
        if review:
            cleaned = clean_review(review)
            vector = vectorizer.transform([cleaned]).toarray()
            result = model.predict(vector)
            
            if result[0] == 1:
                st.success("It's a Positive Review! 🎉")
                st_lottie(lottie_positive, height=300)
            else:
                st.error("It's a Negative Review. 😞")
                st_lottie(lottie_negative, height=300)
        else:
            st.warning("Please enter a review to predict.")

    if st.button("⬅️ Back to Home"):
        st.session_state.page = 'home'
        st.rerun()

