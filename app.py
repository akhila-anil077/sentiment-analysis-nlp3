import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load model, vectorizer, and label encoder
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model, tfidf, le = pickle.load(f)
    return model, tfidf, le

model, tfidf, le = load_model()

# Streamlit Interface
st.title("Sentiment Analysis for Product Reviews")
st.write("Analyze single reviews")

# Text Input
text_input = st.text_area("Enter a review:", "")

if st.button("Predict Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(text_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)
        label = le.inverse_transform(pred)
        st.success(f"Predicted Sentiment: **{label[0].capitalize()}**")

# File Upload
uploaded_file = st.file_uploader("Or upload a CSV file with a 'review' column", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    if 'review' not in df_input.columns:
        st.error("CSV must contain a 'review' column.")
    else:
        df_input['cleaned'] = df_input['review'].astype(str).apply(preprocess)
        vecs = tfidf.transform(df_input['cleaned'])
        preds = model.predict(vecs)
        df_input['predicted_sentiment'] = le.inverse_transform(preds)

        st.subheader("Sentiment Distribution")
        st.bar_chart(df_input['predicted_sentiment'].value_counts())

        st.subheader("Sample Predictions")
        st.dataframe(df_input[['review', 'predicted_sentiment']].head(10))

        # WordClouds
        st.subheader("WordClouds by Sentiment")
        for sentiment in df_input['predicted_sentiment'].unique():
            text = " ".join(df_input[df_input['predicted_sentiment'] == sentiment]['cleaned'])
            if text:
                wc = WordCloud(width=800, height=300, background_color='white').generate(text)
                st.markdown(f"**{sentiment.capitalize()}**")
                st.image(wc.to_array())
