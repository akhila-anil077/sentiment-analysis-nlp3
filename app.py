import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load model components
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')
le = joblib.load('label_encoder.pkl')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
negations = {'no', 'not', 'never', "n't"}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words or word in negations]
    return " ".join(tokens)

# Streamlit UI
st.title("📝 Product Review Sentiment Analyzer")
st.write("Enter a product review below and get the predicted sentiment.")

user_input = st.text_area("Enter your review:", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = tfidf.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        st.success(f"Predicted Sentiment: **{prediction.capitalize()}**")
