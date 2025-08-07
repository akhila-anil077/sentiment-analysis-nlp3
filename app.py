# app.py
import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing - Corrected to match the notebook
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Handle negation
    tokens = text.split()
    negation_handled_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == 'not' and i + 1 < len(tokens):
            negation_handled_tokens.append('not_' + tokens[i+1])
            i += 2
        else:
            negation_handled_tokens.append(tokens[i])
            i += 1
            
    tokens = [lemmatizer.lemmatize(word) for word in negation_handled_tokens if word not in stop_words]
    return ' '.join(tokens)

# Load model pipeline and label encoder
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            pipeline, le = pickle.load(f)
        return pipeline, le
    except FileNotFoundError:
        st.error("model.pkl not found. Please ensure the trained model file is in the same directory.")
        return None, None

pipeline, le = load_model()

if pipeline is not None:
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
            pred = pipeline.predict([cleaned])
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
            preds = pipeline.predict(df_input['cleaned'])
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
